//
//  ToolSupport.swift
//  OnDeviceCatalyst
//
//  Tool calling support for local LLMs (Qwen3, etc.)
//

import Foundation

// MARK: - Tool Definition

/// Defines a tool that the model can call
public struct CatalystTool: Codable, Hashable, Identifiable {
    public let id: String
    public let name: String
    public let description: String
    public let parameters: [CatalystToolParameter]

    public init(id: String? = nil, name: String, description: String, parameters: [CatalystToolParameter] = []) {
        self.id = id ?? name
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// Parameter definition for a tool
public struct CatalystToolParameter: Codable, Hashable {
    public let name: String
    public let type: String
    public let description: String
    public let required: Bool
    public let enumValues: [String]?

    public init(name: String, type: String = "string", description: String, required: Bool = true, enumValues: [String]? = nil) {
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.enumValues = enumValues
    }
}

// MARK: - Tool Invocation (parsed from model output)

/// Represents a tool call parsed from model output
public struct CatalystToolCall: Codable, Identifiable {
    public let id: String
    public let name: String
    public let arguments: [String: AnyCodable]

    public init(id: String = UUID().uuidString, name: String, arguments: [String: AnyCodable]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// Get argument value with type casting
    public func argument<T>(_ key: String) -> T? {
        return arguments[key]?.value as? T
    }
}

// MARK: - Tool Call Parser

/// Parses tool calls from model output
public struct ToolCallParser {

    /// Parse tool calls from text using multiple formats
    /// Supports: Qwen3 native format, JSON code blocks, raw JSON
    public static func parse(from text: String) -> (text: String, toolCalls: [CatalystToolCall]) {
        var cleanText = text
        var toolCalls: [CatalystToolCall] = []

        // 1. Try Qwen3 native format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        let qwenPattern = #"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>"#
        if let regex = try? NSRegularExpression(pattern: qwenPattern, options: []) {
            let range = NSRange(text.startIndex..., in: text)
            let matches = regex.matches(in: text, options: [], range: range)

            for match in matches.reversed() {
                if let jsonRange = Range(match.range(at: 1), in: text) {
                    let jsonString = String(text[jsonRange])
                    if let toolCall = parseJSON(jsonString) {
                        toolCalls.insert(toolCall, at: 0)
                    }
                }
                if let fullRange = Range(match.range, in: cleanText) {
                    cleanText.removeSubrange(fullRange)
                }
            }
        }

        // 2. Try JSON code block format: ```json\n{"tool": "...", "arguments": {...}}\n```
        let codeBlockPattern = #"```json\s*(\{[\s\S]*?\})\s*```"#
        if let regex = try? NSRegularExpression(pattern: codeBlockPattern, options: []) {
            let range = NSRange(cleanText.startIndex..., in: cleanText)
            let matches = regex.matches(in: cleanText, options: [], range: range)

            for match in matches.reversed() {
                if let jsonRange = Range(match.range(at: 1), in: cleanText) {
                    let jsonString = String(cleanText[jsonRange])
                    if let toolCall = parseJSONWithToolKey(jsonString) {
                        toolCalls.insert(toolCall, at: 0)
                    }
                }
                if let fullRange = Range(match.range, in: cleanText) {
                    cleanText.removeSubrange(fullRange)
                }
            }
        }

        return (cleanText.trimmingCharacters(in: .whitespacesAndNewlines), toolCalls)
    }

    /// Parse Qwen3 native format: {"name": "tool_name", "arguments": {...}}
    private static func parseJSON(_ jsonString: String) -> CatalystToolCall? {
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = json["name"] as? String else {
            return nil
        }

        let arguments = (json["arguments"] as? [String: Any]) ?? [:]
        let codableArgs = arguments.mapValues { AnyCodable($0) }

        return CatalystToolCall(name: name, arguments: codableArgs)
    }

    /// Parse seeme-arch format: {"tool": "tool_name", "arguments": {...}}
    private static func parseJSONWithToolKey(_ jsonString: String) -> CatalystToolCall? {
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = json["tool"] as? String else {
            return nil
        }

        let arguments = (json["arguments"] as? [String: Any]) ?? [:]
        let codableArgs = arguments.mapValues { AnyCodable($0) }

        return CatalystToolCall(name: name, arguments: codableArgs)
    }
}

// MARK: - Tool Prompt Formatter

/// Formats tools for inclusion in prompts
public struct ToolPromptFormatter {

    /// Format tools for Qwen3 native tool calling
    public static func formatForQwen3(_ tools: [CatalystTool]) -> String {
        guard !tools.isEmpty else { return "" }

        var prompt = "\n\n# Tools\n\n"
        prompt += "You may call one or more functions to assist with the user query.\n\n"
        prompt += "You are provided with function signatures within <tools></tools> XML tags:\n"
        prompt += "<tools>\n"

        for tool in tools {
            var params: [String: Any] = [:]
            var required: [String] = []

            for param in tool.parameters {
                var paramDef: [String: Any] = [
                    "type": param.type,
                    "description": param.description
                ]
                if let enumVals = param.enumValues {
                    paramDef["enum"] = enumVals
                }
                params[param.name] = paramDef
                if param.required {
                    required.append(param.name)
                }
            }

            let schema: [String: Any] = [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": [
                        "type": "object",
                        "properties": params,
                        "required": required
                    ]
                ]
            ]

            if let jsonData = try? JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys]),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                prompt += jsonString + "\n"
            }
        }

        prompt += "</tools>\n\n"
        prompt += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        prompt += "<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n</tool_call>\n"

        return prompt
    }

    /// Format tools as plain text instructions (works with any model)
    public static func formatAsText(_ tools: [CatalystTool]) -> String {
        guard !tools.isEmpty else { return "" }

        var prompt = "\n\n# AVAILABLE TOOLS\n\n"
        prompt += "You have access to the following tools. Use them when appropriate:\n\n"

        for tool in tools {
            prompt += "## \(tool.name)\n"
            prompt += "\(tool.description)\n\n"
            prompt += "Parameters:\n"

            for param in tool.parameters {
                let requiredFlag = param.required ? "(required)" : "(optional)"
                prompt += "- \(param.name) (\(param.type)) \(requiredFlag): \(param.description)\n"

                if let enumValues = param.enumValues {
                    prompt += "  Options: \(enumValues.joined(separator: ", "))\n"
                }
            }
            prompt += "\n"
        }

        prompt += """
        ## HOW TO USE TOOLS

        When you need to use a tool, output a JSON object in this format:
        ```json
        {
          "tool": "tool_name",
          "arguments": {
            "param1": "value1"
          }
        }
        ```

        Do NOT include any other text before or after the JSON when calling a tool.

        """

        return prompt
    }
}

// MARK: - AnyCodable Helper

/// Type-erased Codable wrapper for tool arguments
public struct AnyCodable: Codable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case let int as Int: try container.encode(int)
        case let double as Double: try container.encode(double)
        case let string as String: try container.encode(string)
        case let bool as Bool: try container.encode(bool)
        case let array as [Any]: try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]: try container.encode(dict.mapValues { AnyCodable($0) })
        default: try container.encodeNil()
        }
    }
}

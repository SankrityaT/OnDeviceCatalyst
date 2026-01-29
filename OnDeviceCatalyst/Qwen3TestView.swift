//
//  Qwen3TestView.swift
//  OnDeviceCatalyst
//
//  Full-featured test view for Qwen3-4B with SOTA capabilities
//

import SwiftUI

struct Qwen3TestView: View {
    // Download state
    @State private var downloadProgress: Double = 0
    @State private var downloadedBytes: Int64 = 0
    @State private var totalBytes: Int64 = 0
    @State private var isDownloading = false
    @State private var downloadComplete = false
    @State private var modelPath: String?

    // Model state
    @State private var isModelLoaded = false
    @State private var isLoading = false
    @State private var loadProgress: String = ""

    // Chat state
    @State private var conversation = Conversation(
        title: "Qwen3 Test",
        systemPrompt: "You are Qwen, a helpful and knowledgeable AI assistant created by Alibaba. Be concise, accurate, and helpful. /no_think"
    )
    @State private var userInput = ""
    @State private var isGenerating = false
    @State private var currentResponse = ""

    // Performance metrics
    @State private var tokensPerSecond: Double = 0
    @State private var totalTokens: Int = 0
    @State private var generationTimeMs: Double = 0

    // Settings
    @State private var showSettings = false
    @State private var selectedPreset: SamplingPreset = .balanced
    @State private var systemPrompt = "You are Qwen, a helpful and knowledgeable AI assistant created by Alibaba. Be concise, accurate, and helpful. /no_think"

    // Error state
    @State private var errorMessage = ""

    /// Filters thinking content from streaming response for display
    private var filteredResponse: String {
        var content = currentResponse

        // Remove <think>...</think> blocks
        if let thinkStart = content.range(of: "<think>") {
            if let thinkEnd = content.range(of: "</think>") {
                // Full think block - remove it (upperBound is already exclusive)
                content.removeSubrange(thinkStart.lowerBound..<thinkEnd.upperBound)
            } else {
                // Partial think block - hide everything from <think> onwards
                content = String(content[..<thinkStart.lowerBound])
            }
        }

        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Model profile
    @State private var modelProfile: ModelProfile?

    // Qwen3-4B - latest architecture with thinking capabilities
    private let modelURL = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    private let modelFileName = "Qwen3-4B-Q4_K_M.gguf"

    enum SamplingPreset: String, CaseIterable {
        case deterministic = "Deterministic"
        case balanced = "Balanced"
        case creative = "Creative"
        case mirostat = "Mirostat v2"

        var config: PredictionConfig {
            switch self {
            case .deterministic: return .deterministic
            case .balanced: return .balanced
            case .creative: return .creative
            case .mirostat: return .mirostat
            }
        }

        var description: String {
            switch self {
            case .deterministic: return "Greedy, consistent outputs"
            case .balanced: return "Good mix of quality & variety"
            case .creative: return "More varied, creative responses"
            case .mirostat: return "Dynamic entropy targeting"
            }
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if !downloadComplete {
                    downloadSection
                } else if !isModelLoaded {
                    loadSection
                } else {
                    chatSection
                }

                // Error display
                if !errorMessage.isEmpty {
                    errorBanner
                }
            }
            .navigationTitle("Qwen3 Test")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if isModelLoaded {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Clear") {
                            clearChat()
                        }
                    }
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: { showSettings.toggle() }) {
                            Image(systemName: "slider.horizontal.3")
                        }
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                settingsSheet
            }
        }
        .onAppear {
            checkExistingModel()
        }
    }

    // MARK: - Download Section

    private var downloadSection: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "arrow.down.circle")
                .font(.system(size: 60))
                .foregroundColor(.blue)

            Text("Download Qwen3-4B")
                .font(.title2)
                .fontWeight(.bold)

            Text("Q4_K_M quantization (~2.7 GB)")
                .foregroundColor(.secondary)

            if isDownloading {
                VStack(spacing: 12) {
                    ProgressView(value: downloadProgress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding(.horizontal, 40)

                    Text("\(formatBytes(downloadedBytes)) / \(formatBytes(totalBytes))")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("\(Int(downloadProgress * 100))%")
                        .font(.headline)
                }
            } else {
                Button(action: startDownload) {
                    Label("Download Model", systemImage: "arrow.down.circle.fill")
                        .frame(maxWidth: .infinity)
                        .padding()
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal, 40)
            }

            Spacer()
        }
        .padding()
    }

    // MARK: - Load Section

    private var loadSection: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "cpu")
                .font(.system(size: 60))
                .foregroundColor(.green)

            Text("Model Downloaded")
                .font(.title2)
                .fontWeight(.bold)

            Text(formatBytes(totalBytes))
                .foregroundColor(.secondary)

            if isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text(loadProgress)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                Button(action: loadModel) {
                    Label("Load Model", systemImage: "play.circle.fill")
                        .frame(maxWidth: .infinity)
                        .padding()
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal, 40)
            }

            Spacer()
        }
        .padding()
    }

    // MARK: - Chat Section

    private var chatSection: some View {
        VStack(spacing: 0) {
            // Performance bar
            if totalTokens > 0 {
                performanceBar
            }

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(conversation.turns.filter { $0.role != .system }) { turn in
                            Qwen3ChatBubble(turn: turn)
                        }

                        if isGenerating && !filteredResponse.isEmpty {
                            Qwen3ChatBubble(turn: Turn.assistant(filteredResponse))
                                .id("generating")
                        }

                        if isGenerating && filteredResponse.isEmpty {
                            HStack {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Thinking...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .id("thinking")
                        }
                    }
                    .padding()
                }
                .onChange(of: conversation.turns.count) { _ in
                    withAnimation {
                        proxy.scrollTo(conversation.turns.last?.id, anchor: .bottom)
                    }
                }
                .onChange(of: currentResponse) { _ in
                    withAnimation {
                        proxy.scrollTo("generating", anchor: .bottom)
                    }
                }
            }

            Divider()

            // Input area
            inputArea
        }
    }

    private var performanceBar: some View {
        HStack {
            Label("\(String(format: "%.1f", tokensPerSecond)) tok/s", systemImage: "speedometer")
            Spacer()
            Text("\(totalTokens) tokens")
            Spacer()
            Text("\(Int(generationTimeMs))ms")
        }
        .font(.caption)
        .foregroundColor(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color.gray.opacity(0.1))
    }

    private var inputArea: some View {
        VStack(spacing: 8) {
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message Qwen3...", text: $userInput, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(20)
                    .lineLimit(1...5)
                    .disabled(isGenerating)

                if isGenerating {
                    Button(action: stopGeneration) {
                        Image(systemName: "stop.circle.fill")
                            .font(.title2)
                            .foregroundColor(.red)
                    }
                } else {
                    Button(action: sendMessage) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundColor(userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .gray : .blue)
                    }
                    .disabled(userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }

            // Quick prompts
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    Qwen3QuickPromptButton(text: "Explain quantum computing") { userInput = $0 }
                    Qwen3QuickPromptButton(text: "Write a haiku about AI") { userInput = $0 }
                    Qwen3QuickPromptButton(text: "What's 2+2? Think step by step") { userInput = $0 }
                    Qwen3QuickPromptButton(text: "Tell me a short joke") { userInput = $0 }
                }
            }
        }
        .padding()
    }

    // MARK: - Settings Sheet

    private var settingsSheet: some View {
        NavigationView {
            Form {
                Section("Sampling Mode") {
                    Picker("Preset", selection: $selectedPreset) {
                        ForEach(SamplingPreset.allCases, id: \.self) { preset in
                            Text(preset.rawValue).tag(preset)
                        }
                    }
                    .pickerStyle(.segmented)

                    Text(selectedPreset.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("System Prompt") {
                    TextEditor(text: $systemPrompt)
                        .frame(minHeight: 100)
                }

                Section("Test Scenarios") {
                    Button("Test Long Context") {
                        testLongContext()
                        showSettings = false
                    }

                    Button("Test Reasoning") {
                        testReasoning()
                        showSettings = false
                    }

                    Button("Test Code Generation") {
                        testCodeGeneration()
                        showSettings = false
                    }

                    Button("Test Creative Writing") {
                        testCreativeWriting()
                        showSettings = false
                    }
                }

                Section("Info") {
                    LabeledContent("Model", value: "Qwen3-4B-Instruct")
                    LabeledContent("Architecture", value: "qwen3")
                    LabeledContent("Quantization", value: "Q4_K_M (4-bit)")
                    LabeledContent("Size", value: "~2.7 GB")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        conversation.systemPrompt = systemPrompt
                        showSettings = false
                    }
                }
            }
        }
    }

    // MARK: - Error Banner

    private var errorBanner: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            Text(errorMessage)
                .font(.caption)
            Spacer()
            Button("Dismiss") {
                errorMessage = ""
            }
            .font(.caption)
        }
        .padding()
        .background(Color.red.opacity(0.1))
    }

    // MARK: - Actions

    private func checkExistingModel() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let filePath = documentsPath.appendingPathComponent(modelFileName)

        if FileManager.default.fileExists(atPath: filePath.path) {
            modelPath = filePath.path
            downloadComplete = true

            if let attrs = try? FileManager.default.attributesOfItem(atPath: filePath.path),
               let size = attrs[.size] as? Int64 {
                totalBytes = size
            }
        }
    }

    private func startDownload() {
        isDownloading = true
        errorMessage = ""
        totalBytes = 2_700_000_000

        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destinationURL = documentsPath.appendingPathComponent(modelFileName)

        guard let url = URL(string: modelURL) else {
            errorMessage = "Invalid URL"
            isDownloading = false
            return
        }

        let delegate = Qwen3DownloadDelegate(
            destinationURL: destinationURL,
            onProgress: { _, totalBytesWritten, totalBytesExpected in
                DispatchQueue.main.async {
                    self.downloadedBytes = totalBytesWritten
                    if totalBytesExpected > 0 {
                        self.totalBytes = totalBytesExpected
                    }
                    self.downloadProgress = Double(totalBytesWritten) / Double(self.totalBytes)
                }
            },
            onComplete: { savedURL, error in
                DispatchQueue.main.async {
                    self.isDownloading = false

                    if let error = error {
                        self.errorMessage = "Download failed: \(error.localizedDescription)"
                        return
                    }

                    guard let savedURL = savedURL else {
                        self.errorMessage = "Download failed: No file saved"
                        return
                    }

                    if let attrs = try? FileManager.default.attributesOfItem(atPath: savedURL.path),
                       let size = attrs[.size] as? Int64 {
                        self.totalBytes = size
                        self.downloadedBytes = size
                    }

                    self.modelPath = savedURL.path
                    self.downloadComplete = true
                    self.downloadProgress = 1.0
                }
            }
        )

        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: .main)
        let task = session.downloadTask(with: url)
        objc_setAssociatedObject(task, "delegate", delegate, .OBJC_ASSOCIATION_RETAIN)
        task.resume()
    }

    private func loadModel() {
        guard let path = modelPath else {
            errorMessage = "No model path"
            return
        }

        isLoading = true
        loadProgress = "Initializing..."
        errorMessage = ""

        Task {
            do {
                loadProgress = "Creating profile..."

                let profile = try ModelProfile(
                    filePath: path,
                    name: "Qwen3-4B",
                    architecture: ModelArchitecture.qwen3
                )

                loadProgress = "Loading into memory..."

                let (_, loadStream) = await Catalyst.shared.instance(
                    for: profile,
                    settings: .balanced,
                    predictionConfig: selectedPreset.config
                )

                for await progress in loadStream {
                    await MainActor.run {
                        loadProgress = progress.message
                    }

                    if progress.isComplete {
                        if case .failed(let message) = progress {
                            throw CatalystError.modelLoadingFailed(details: message)
                        }
                        break
                    }
                }

                await MainActor.run {
                    modelProfile = profile
                    isModelLoaded = true
                    isLoading = false
                }

            } catch {
                await MainActor.run {
                    errorMessage = "Load failed: \(error.localizedDescription)"
                    isLoading = false
                }
            }
        }
    }

    private func sendMessage() {
        let message = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty, let profile = modelProfile else { return }

        userInput = ""
        isGenerating = true
        currentResponse = ""
        errorMessage = ""

        // Add user message
        conversation.addTurn(Turn.user(message))

        Task {
            do {
                let startTime = Date()
                var tokenCount = 0

                let stream = try await Catalyst.shared.generate(
                    conversation: conversation.turns,
                    systemPrompt: conversation.systemPrompt ?? systemPrompt,
                    using: profile,
                    settings: .balanced,
                    predictionConfig: selectedPreset.config
                )

                for try await chunk in stream {
                    tokenCount += 1

                    await MainActor.run {
                        currentResponse += chunk.content
                        totalTokens = tokenCount

                        let elapsed = Date().timeIntervalSince(startTime)
                        if elapsed > 0 {
                            tokensPerSecond = Double(tokenCount) / elapsed
                            generationTimeMs = elapsed * 1000
                        }
                    }

                    if chunk.isComplete {
                        break
                    }
                }

                await MainActor.run {
                    // Add assistant response to conversation (filtered)
                    let finalResponse = filteredResponse
                    if !finalResponse.isEmpty {
                        conversation.addTurn(Turn.assistant(finalResponse))
                    }
                    currentResponse = ""
                    isGenerating = false
                }

            } catch {
                await MainActor.run {
                    errorMessage = "Generation failed: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }

    private func stopGeneration() {
        if let profile = modelProfile {
            Catalyst.shared.interruptGeneration(for: profile.id)
        }
        isGenerating = false
    }

    private func clearChat() {
        conversation = Conversation(title: "Qwen3 Test", systemPrompt: systemPrompt)
        currentResponse = ""
        totalTokens = 0
        tokensPerSecond = 0
        generationTimeMs = 0
    }

    // MARK: - Test Scenarios

    private func testLongContext() {
        userInput = """
        Here's a complex scenario: A company has 500 employees across 5 departments. Marketing has 100 employees with average salary $60,000. Engineering has 150 employees with average salary $95,000. Sales has 120 employees with average salary $70,000. HR has 50 employees with average salary $55,000. Finance has 80 employees with average salary $75,000.

        Calculate: 1) Total annual payroll 2) Average salary across the company 3) Which department has the highest total payroll 4) What percentage of employees are in Engineering?
        """
    }

    private func testReasoning() {
        userInput = "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Think through this step by step before answering."
    }

    private func testCodeGeneration() {
        userInput = "Write a Swift function that implements binary search on a sorted array. Include comments explaining the algorithm."
    }

    private func testCreativeWriting() {
        selectedPreset = .creative
        userInput = "Write a short story (100 words) about a robot discovering emotions for the first time."
    }

    // MARK: - Helpers

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

// MARK: - Chat Bubble

struct Qwen3ChatBubble: View {
    let turn: Turn

    /// Filters out Qwen3 thinking content (<think>...</think>) from the response
    private var displayContent: String {
        var content = turn.content

        // Remove <think>...</think> blocks (including partial ones during streaming)
        if let thinkStart = content.range(of: "<think>") {
            if let thinkEnd = content.range(of: "</think>") {
                // Full think block - remove it entirely (upperBound is already exclusive)
                content.removeSubrange(thinkStart.lowerBound..<thinkEnd.upperBound)
            } else {
                // Partial think block (still streaming) - hide everything from <think> onwards
                content = String(content[..<thinkStart.lowerBound])
            }
        }

        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        HStack {
            if turn.role == .user { Spacer() }

            VStack(alignment: turn.role == .user ? .trailing : .leading, spacing: 4) {
                Text(displayContent)
                    .padding(12)
                    .background(turn.role == .user ? Color.blue : Color.gray.opacity(0.2))
                    .foregroundColor(turn.role == .user ? .white : .primary)
                    .cornerRadius(16)

                if let metadata = turn.metadata, let tps = metadata.tokensPerSecond {
                    Text("\(String(format: "%.1f", tps)) tok/s")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            #if os(iOS)
            .frame(maxWidth: UIScreen.main.bounds.width * 0.75, alignment: turn.role == .user ? .trailing : .leading)
            #else
            .frame(maxWidth: 500, alignment: turn.role == .user ? .trailing : .leading)
            #endif

            if turn.role == .assistant { Spacer() }
        }
    }
}

// MARK: - Quick Prompt Button

struct Qwen3QuickPromptButton: View {
    let text: String
    let action: (String) -> Void

    var body: some View {
        Button(action: { action(text) }) {
            Text(text)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.gray.opacity(0.15))
                .cornerRadius(16)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Download Delegate

class Qwen3DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let destinationURL: URL
    let onProgress: (Int64, Int64, Int64) -> Void
    let onComplete: (URL?, Error?) -> Void

    init(
        destinationURL: URL,
        onProgress: @escaping (Int64, Int64, Int64) -> Void,
        onComplete: @escaping (URL?, Error?) -> Void
    ) {
        self.destinationURL = destinationURL
        self.onProgress = onProgress
        self.onComplete = onComplete
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.moveItem(at: location, to: destinationURL)
            onComplete(destinationURL, nil)
        } catch {
            onComplete(nil, error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            onComplete(nil, error)
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(bytesWritten, totalBytesWritten, totalBytesExpectedToWrite)
    }
}

// MARK: - Preview

struct Qwen3TestView_Previews: PreviewProvider {
    static var previews: some View {
        Qwen3TestView()
    }
}

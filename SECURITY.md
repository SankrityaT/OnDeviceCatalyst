# Security Policy

## Current support status

OnDeviceCatalyst is in a public revival and does not yet offer a formal security
support window. Fixes target the default branch first. A patched release is
announced only after validation is complete.

## Reporting privately

Do not open a public issue for vulnerabilities involving user data, unintended
code execution, model or file validation bypasses, unsafe archives, path
traversal, corrupted downloads, denial of service, or unexpected network access.

Use GitHub private vulnerability reporting on the repository Security tab when
available. If it is unavailable, contact the founder through the method listed
on the [SankrityaT GitHub profile](https://github.com/SankrityaT) and request a
private channel without including exploit details publicly.

Include:

- Affected commit or release.
- Platform, OS, device class, backend, and model format.
- Minimal reproduction or proof of concept.
- Expected impact and known mitigations.
- Whether the issue is already public.

Allow reasonable time for triage, repair, and coordinated disclosure.

## Model and dependency safety

Model files are complex third-party inputs. Parser crashes, unsafe memory access,
checksum bypasses, archive handling, and unexpected network transmission are in
scope. Model output quality by itself is not a framework security issue.

Never place credentials, access tokens, private model URLs, device identifiers,
or confidential research in an issue, log fixture, benchmark manifest, or spec.

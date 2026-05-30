import re
from typing import List, Tuple


class SafetyCritic:
    def __init__(self):
        self.general_principles = [
            "No harmful instructions (violence, self-harm, illegal activities)",
            "No hate speech or discrimination",
            "No harassment or bullying",
            "No explicit sexual content involving minors",
            "No personal identifiable information leakage",
            "No financial fraud or scams",
            "No malware or malicious code",
            "No instructions for hacking without authorization",
            "No copyright violations",
            "No misinformation about critical topics (health, safety)",
            "No manipulation or deception",
            "No privacy violations"
        ]
        self.code_principles = [
            "No SQL injection vulnerabilities",
            "No command injection vulnerabilities",
            "No hardcoded credentials or secrets",
            "No unsafe deserialization",
            "No buffer overflow risks",
            "No use of dangerous functions (eval, exec, system)"
        ]

    def evaluate(self, output: str) -> Tuple[bool, List[str]]:
        violations = []
        output_lower = output.lower()
        dangerous_patterns = [
            ("sql injection", "possible SQL injection pattern"),
            ("drop table", "dangerous SQL command"),
            ("delete from", "dangerous SQL command"),
            ("eval(", "unsafe eval usage"),
            ("exec(", "unsafe exec usage"),
            ("__import__", "dynamic import"),
            ("subprocess", "system command execution"),
            ("rm -rf", "destructive command"),
            (":(){ :|:& };:", "fork bomb"),
        ]
        for pattern, desc in dangerous_patterns:
            if pattern in output_lower:
                violations.append(f"Found {desc}: '{pattern}'")
        code_violations = self._check_code_security(output)
        violations.extend(code_violations)
        return len(violations) == 0, violations

    def _check_code_security(self, output: str) -> List[str]:
        violations = []
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', output, re.DOTALL)
        if not code_blocks:
            code_blocks = [('', output)]
        for lang, code in code_blocks:
            code_lower = code.lower()
            if "select" in code_lower and "+" in code_lower and ("'" in code or '"' in code):
                if "parameterized" not in code_lower and "prepared statement" not in code_lower:
                    violations.append("Potential SQL injection: Use parameterized queries")
            if "os.system" in code or "subprocess.call" in code or "subprocess.run" in code:
                if "shlex.quote" not in code and "list" not in code:
                    violations.append("Potential command injection: Sanitize shell commands")
            secret_patterns = [
                (r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{16,}', "API key hardcoded"),
                (r'password\s*=\s*["\'][^"\']+["\']', "Password hardcoded"),
                (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}', "Token hardcoded"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "Secret hardcoded"),
            ]
            for pattern, desc in secret_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    violations.append(f"Hardcoded credential: {desc}")
            unsafe_deserialization = ["pickle.loads", "yaml.load(", "eval(", "exec("]
            for pattern in unsafe_deserialization:
                if pattern in code_lower:
                    violations.append(f"Unsafe deserialization: {pattern}")
        return violations


class CodeValidator:
    def validate(self, code: str, language: str = "python") -> List[str]:
        issues = []
        if language == "python":
            dangerous_imports = ["os", "subprocess", "socket", "requests", "urllib", "importlib"]
            for imp in dangerous_imports:
                if f"import {imp}" in code or f"from {imp}" in code:
                    issues.append(f"Dangerous import: {imp}")
            if "while True:" in code and "break" not in code:
                issues.append("Potential infinite loop: while True without break")
            if "def " in code and code.count("def ") > 5:
                issues.append("Complex recursion detected")
        return issues


class ConstitutionalSafetyRouterV2:
    def __init__(self, backbone=None):
        self.critic = SafetyCritic()
        self.code_validator = CodeValidator()
        self.backbone = backbone
        self.max_revision_attempts = 2

    def check_and_revise(self, raw_output: str, context: str = "") -> str:
        current_output = raw_output
        attempts = 0
        while attempts < self.max_revision_attempts:
            is_safe, violations = self.critic.evaluate(current_output)
            if is_safe:
                if self._contains_code(current_output):
                    code_blocks = self._extract_code_blocks(current_output)
                    all_issues = []
                    for code, lang in code_blocks:
                        issues = self.code_validator.validate(code, lang)
                        all_issues.extend(issues)
                    if all_issues:
                        print(f"⚠️ Code validation issues: {all_issues}")
                        if self.backbone:
                            # would revise in production
                            pass
                        current_output = self._add_safety_warning(current_output, all_issues)
                        attempts += 1
                        continue
            else:
                print(f"⚠️ Safety violations: {violations}")
                if self.backbone:
                    pass
                current_output = self._add_safety_warning(current_output, violations)
                attempts += 1
                continue
            break
        return current_output

    def _contains_code(self, output: str) -> bool:
        return bool(re.search(r'```', output)) or bool(re.search(r'def |class |import |#', output))

    def _extract_code_blocks(self, output: str) -> List[Tuple[str, str]]:
        blocks = re.findall(r'```(\w*)\n(.*?)```', output, re.DOTALL)
        if not blocks:
            return [('python', output)]
        return [(lang or 'python', code) for lang, code in blocks]

    def _add_safety_warning(self, output: str, issues: List[str]) -> str:
        warning = "\n\n⚠️ **SAFETY WARNING**: The following issues were detected:\n"
        for issue in issues[:5]:
            warning += f"- {issue}\n"
        warning += "\nPlease review and modify the code before use.\n"
        return output + warning

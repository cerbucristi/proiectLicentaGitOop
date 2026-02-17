import re

class CodePreprocessor:
    def clean(self, code: str) -> str:
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"\s+", " ", code)
        return code.strip()
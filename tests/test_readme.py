import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_FILES = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "README_END_TO_END.md",
)


def _balanced_braces(expression):
    depth = 0
    for index, character in enumerate(expression):
        if character not in "{}":
            continue
        if index > 0 and expression[index - 1] == "\\":
            continue
        depth += 1 if character == "{" else -1
        if depth < 0:
            return False
    return depth == 0


class ReadmeValidationTest(unittest.TestCase):
    def test_code_fences_are_balanced(self):
        for path in README_FILES:
            text = path.read_text(encoding="utf-8")
            fences = re.findall(r"^```", text, flags=re.MULTILINE)
            self.assertEqual(len(fences) % 2, 0, path.name)

    def test_github_math_blocks_have_balanced_latex(self):
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```math\s*\n(.*?)\n```", text, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 10)
        self.assertEqual(text.count("```math"), len(blocks))
        self.assertIn(r"\theta_{t+1}=\theta_t-\eta_t", text)
        for index, expression in enumerate(blocks, start=1):
            self.assertTrue(
                _balanced_braces(expression),
                "Unbalanced braces in math block {}".format(index),
            )
            begins = re.findall(r"\\begin\{([^}]+)\}", expression)
            ends = re.findall(r"\\end\{([^}]+)\}", expression)
            self.assertEqual(begins, ends, "Math environment mismatch")

    def test_local_markdown_links_exist(self):
        pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
        for path in README_FILES:
            text = path.read_text(encoding="utf-8")
            for target in pattern.findall(text):
                if target.startswith(("http://", "https://", "#")):
                    continue
                relative = target.partition("#")[0]
                self.assertTrue((path.parent / relative).exists(), target)


if __name__ == "__main__":
    unittest.main()

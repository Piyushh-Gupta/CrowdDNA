import os

def test_documentation_exists():
    required_docs = [
        "docs/TEAM.md",
        "docs/ONBOARDING.md",
        "docs/WORKFLOW.md",
        "docs/OWNERSHIP.md",
        "docs/CODE_REVIEW.md",
        "docs/BRANCHING.md",
        "docs/CONTRIBUTOR_CHECKLIST.md",
        "docs/TASK_ASSIGNMENT.md",
        "docs/PROJECT_MANAGEMENT.md",
        "docs/GITHUB_PROJECT_SETUP.md",
        "docs/DEFINITION_OF_DONE.md",
        "docs/ADR/ADR-0003.md",
        "ROADMAP.md",
        "CODEOWNERS"
    ]
    for doc in required_docs:
        assert os.path.exists(doc), f"Missing required document: {doc}"

def test_ownership_exists():
    with open("docs/OWNERSHIP.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Technical Lead: Piyush Gupta" in content
        assert "Core Contributor: Aayushi" in content

def test_adr_numbering():
    with open("docs/ADR/ADR-0003.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "ADR 0003" in content

def test_markdown_links():
    # Simple check for markdown link structure in workflow
    with open("docs/WORKFLOW.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "```mermaid" in content
        assert "Idea --> Issue" in content

def test_required_sections():
    with open("docs/ONBOARDING.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "## Setup" in content
        assert "## First Contribution Walkthrough" in content

def test_roadmap_ownership():
    with open("ROADMAP.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "**Owner**:" in content
        assert "**Reviewer**:" in content

def test_definition_of_done_presence():
    with open("docs/DEFINITION_OF_DONE.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Ruff passes" in content
        assert "Tests pass" in content
        assert "Reviewer approval" in content

def test_github_documentation_presence():
    assert os.path.exists(".github/ISSUE_TEMPLATE/feature_request.md")
    assert os.path.exists(".github/ISSUE_TEMPLATE/bug_report.md")
    assert os.path.exists(".github/PULL_REQUEST_TEMPLATE.md")

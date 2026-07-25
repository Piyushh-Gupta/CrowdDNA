import os
import re
from pathlib import Path

def test_required_docs_exist():
    required_files = [
        "docs/GOVERNANCE.md",
        "docs/CONTRIBUTING.md",
        "docs/DEVELOPMENT.md",
        "docs/ROADMAP.md",
        "docs/PROJECT_STRUCTURE.md",
        "docs/PROJECT_STATUS.md",
        "docs/DECISION_LOG.md",
        "docs/CODING_STANDARDS.md",
        "docs/ARCHITECTURE_OVERVIEW.md",
        "docs/DEPENDENCY_GRAPH.md",
        "docs/API_REFERENCE.md",
        "docs/RUNBOOK.md",
        "docs/TESTING_GUIDE.md",
        "docs/RELEASE_PROCESS.md",
        "docs/SECURITY_POLICY.md",
        "docs/SUPPORT.md",
        "docs/FAQ.md",
        "docs/GLOSSARY.md",
        "docs/STYLE_GUIDE.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/CODEOWNERS",
        ".github/SECURITY.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/ISSUE_TEMPLATE/question.md",
        "docs/adr/ADR-0001-use-architecture-decision-records.md",
        "docs/adr/ADR-0002-decouple-release-from-deployment.md"
    ]
    
    for f in required_files:
        assert os.path.exists(f), f"Missing required file: {f}"

def test_adr_numbering():
    adr_dir = Path("docs/adr")
    assert adr_dir.exists()
    files = list(adr_dir.glob("ADR-*.md"))
    assert len(files) >= 2
    
    numbers = []
    for f in files:
        m = re.match(r"ADR-(\d{4})-.*\.md", f.name)
        assert m, f"Invalid ADR filename format: {f.name}"
        numbers.append(int(m.group(1)))
        
    numbers.sort()
    # Check if they are strictly sequential
    assert numbers == list(range(1, len(numbers) + 1)), "ADRs are not sequentially numbered"

def test_cross_references():
    docs_dir = Path("docs")
    md_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("adr/*.md")) + list(Path(".github").glob("*.md"))
    
    link_pattern = re.compile(r'\[.*?\]\(([^http].*?)\)')
    
    for md_file in md_files:
        content = md_file.read_text(encoding='utf-8')
        links = link_pattern.findall(content)
        for link in links:
            target = link.split('#')[0]
            if not target:
                continue
                
            target_path = (md_file.parent / target).resolve()
            assert target_path.exists(), f"Broken link in {md_file}: {link}"
            
def test_required_sections_in_status():
    status_content = Path("docs/PROJECT_STATUS.md").read_text(encoding='utf-8').lower()
    required_sections = [
        "completed phases",
        "current phase",
        "remaining phases",
        "implementation progress",
        "roadmap progress",
        "production readiness",
        "known technical debt",
        "future milestones"
    ]
    for section in required_sections:
        assert section in status_content, f"Missing section '{section}' in PROJECT_STATUS.md"

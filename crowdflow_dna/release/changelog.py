class ChangelogGenerator:
    def generate(self, commits: list):
        # Deduplicate deterministically
        unique_commits = sorted(list(set(commits)))
        
        changelog = {
            "Features": [],
            "Fixes": [],
            "Performance": [],
            "Security": [],
            "Documentation": [],
            "Refactoring": [],
            "Breaking Changes": []
        }
        
        for commit in unique_commits:
            if commit.startswith("feat"):
                changelog["Features"].append(commit)
            elif commit.startswith("fix"):
                changelog["Fixes"].append(commit)
            elif commit.startswith("perf"):
                changelog["Performance"].append(commit)
            elif commit.startswith("sec"):
                changelog["Security"].append(commit)
            elif commit.startswith("docs"):
                changelog["Documentation"].append(commit)
            elif commit.startswith("refactor"):
                changelog["Refactoring"].append(commit)
                
            if "BREAKING CHANGE" in commit:
                changelog["Breaking Changes"].append(commit)
            
        markdown = "# Changelog\n"
        for section, msgs in changelog.items():
            if msgs:
                markdown += f"\n## {section}\n"
                for m in sorted(msgs):
                    markdown += f"- {m}\n"
        return markdown

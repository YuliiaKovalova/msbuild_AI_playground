import json
import os
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import difflib
from pathlib import Path

class PRDiffAnalyzer:
    """Analyze PR diffs to identify potential bugs and risky changes"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, 'raw-data', 'msbuild')
        self.prs_dir = os.path.join(self.raw_data_dir, 'prs')
        self.diffs_dir = os.path.join(self.raw_data_dir, 'diffs')
        
        # Bug pattern categories
        self.bug_patterns = {
            'null_checks': {
                'patterns': [
                    r'-\s*if\s*\([^)]*!=\s*null[^)]*\)',  # Removed null checks
                    r'-\s*if\s*\([^)]*null\s*!=[^)]*\)',
                    r'\+[^/]*\.\w+\s*\(',  # Added method calls without null checks
                    r'-\s*\?\.',  # Removed null-conditional operators
                ],
                'risk': 'high',
                'description': 'Removed null checks or unsafe dereferences'
            },
            'exception_handling': {
                'patterns': [
                    r'-\s*try\s*{',  # Removed try blocks
                    r'-\s*catch\s*\([^)]*\)',  # Removed catch blocks
                    r'\+\s*throw\s+new',  # Added exceptions
                    r'-\s*finally\s*{',  # Removed finally blocks
                ],
                'risk': 'high',
                'description': 'Changes to exception handling'
            },
            'concurrency': {
                'patterns': [
                    r'-\s*lock\s*\(',  # Removed locks
                    r'-\s*synchronized',
                    r'\+\s*Task\.Run',  # Added async operations
                    r'\+\s*async\s+',  # Added async methods
                    r'-\s*volatile\s+',  # Removed volatile
                    r'[+-]\s*Interlocked\.',  # Changes to thread-safe operations
                ],
                'risk': 'critical',
                'description': 'Threading and concurrency changes'
            },
            'resource_management': {
                'patterns': [
                    r'-\s*using\s*\(',  # Removed using statements
                    r'-\s*\.Dispose\(\)',  # Removed dispose calls
                    r'-\s*\.Close\(\)',  # Removed close calls
                    r'\+\s*new\s+FileStream',  # Added file operations
                    r'\+\s*new\s+.*Connection',  # Added connections
                ],
                'risk': 'high',
                'description': 'Resource management changes'
            },
            'boundary_conditions': {
                'patterns': [
                    r'[+-]\s*[<>]=',  # Changed comparison operators
                    r'-\s*if\s*\([^)]*\.Length\s*[<>]',  # Removed length checks
                    r'-\s*if\s*\([^)]*\.Count\s*[<>]',  # Removed count checks
                    r'[+-]\s*\[[^\]]*\]',  # Array access changes
                    r'[+-]\s*for\s*\([^;]*;[^;]*[<>]=',  # Loop boundary changes
                ],
                'risk': 'medium',
                'description': 'Boundary condition changes'
            },
            'type_safety': {
                'patterns': [
                    r'-\s*as\s+\w+',  # Removed safe casts
                    r'\+\s*\(\w+\)',  # Added direct casts
                    r'-\s*is\s+\w+',  # Removed type checks
                    r'\+\s*dynamic\s+',  # Added dynamic types
                ],
                'risk': 'medium',
                'description': 'Type safety changes'
            },
            'configuration': {
                'patterns': [
                    r'[+-]\s*"[^"]*"',  # String literal changes
                    r'[+-]\s*\d+',  # Numeric constant changes
                    r'[+-]\s*TimeSpan\.',  # Timeout changes
                    r'[+-]\s*const\s+',  # Constant changes
                ],
                'risk': 'low',
                'description': 'Configuration and constant changes'
            },
            'api_changes': {
                'patterns': [
                    r'-\s*public\s+',  # Removed public members
                    r'\+\s*\[Obsolete',  # Added obsolete attributes
                    r'-\s*virtual\s+',  # Removed virtual
                    r'[+-]\s*interface\s+',  # Interface changes
                ],
                'risk': 'high',
                'description': 'API breaking changes'
            },
            'performance': {
                'patterns': [
                    r'\+\s*\.ToList\(\)',  # Added materializations
                    r'\+\s*\.ToArray\(\)',
                    r'-\s*\.AsParallel\(\)',  # Removed parallelization
                    r'\+\s*foreach.*foreach',  # Nested loops
                    r'\+\s*\.\w+\(\)\.Where\(',  # Added LINQ chains
                ],
                'risk': 'medium',
                'description': 'Potential performance impacts'
            },
            'security': {
                'patterns': [
                    r'\+\s*Password\s*=',  # Hardcoded passwords
                    r'\+\s*"[^"]*password[^"]*"',
                    r'-\s*\[Authorize',  # Removed authorization
                    r'\+\s*AllowAnonymous',  # Added anonymous access
                    r'\+\s*Process\.Start',  # Process execution
                ],
                'risk': 'critical',
                'description': 'Security-related changes'
            }
        }
        
        self.analysis_results = {
            'total_prs': 0,
            'analyzed_prs': 0,
            'risky_prs': [],
            'bug_patterns_found': defaultdict(list),
            'risk_summary': defaultdict(int),
            'large_prs': [],
            'files_with_most_changes': defaultdict(int)
        }

    def load_pr_data(self, pr_number: int) -> Optional[Dict]:
        """Load PR metadata"""
        pr_file = os.path.join(self.prs_dir, f'pr-{pr_number}.json')
        if os.path.exists(pr_file):
            try:
                with open(pr_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading PR {pr_number}: {e}")
        return None

    def load_diff(self, pr_number: int) -> Optional[str]:
        """Load diff content for a PR"""
        diff_file = os.path.join(self.diffs_dir, f'pr-{pr_number}.diff')
        if os.path.exists(diff_file):
            try:
                with open(diff_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error loading diff for PR {pr_number}: {e}")
        return None

    def parse_diff(self, diff_content: str) -> List[Dict]:
        """Parse diff into structured format"""
        files = []
        current_file = None
        
        lines = diff_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # New file
            if line.startswith('diff --git'):
                if current_file:
                    files.append(current_file)
                
                match = re.match(r'diff --git a/(.*) b/(.*)', line)
                if match:
                    current_file = {
                        'old_path': match.group(1),
                        'new_path': match.group(2),
                        'chunks': [],
                        'additions': 0,
                        'deletions': 0
                    }
            
            # Hunk header
            elif line.startswith('@@'):
                if current_file:
                    hunk_match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', line)
                    if hunk_match:
                        current_chunk = {
                            'old_start': int(hunk_match.group(1)),
                            'old_lines': int(hunk_match.group(2) or 1),
                            'new_start': int(hunk_match.group(3)),
                            'new_lines': int(hunk_match.group(4) or 1),
                            'context': hunk_match.group(5).strip(),
                            'changes': []
                        }
                        current_file['chunks'].append(current_chunk)
            
            # Actual changes
            elif current_file and current_file['chunks'] and line and line[0] in ['+', '-', ' ']:
                current_chunk = current_file['chunks'][-1]
                current_chunk['changes'].append(line)
                
                if line.startswith('+') and not line.startswith('+++'):
                    current_file['additions'] += 1
                elif line.startswith('-') and not line.startswith('---'):
                    current_file['deletions'] += 1
            
            i += 1
        
        if current_file:
            files.append(current_file)
        
        return files

    def analyze_diff_for_bugs(self, diff_files: List[Dict], pr_data: Dict) -> Dict:
        """Analyze parsed diff for potential bugs"""
        findings = {
            'pr_number': pr_data['number'],
            'pr_title': pr_data.get('title', 'Unknown'),
            'author': pr_data.get('author', {}).get('login', 'Unknown'),
            'created_at': pr_data.get('createdAt', 'Unknown'),
            'merged': pr_data.get('mergedAt') is not None,
            'total_changes': sum(f['additions'] + f['deletions'] for f in diff_files),
            'files_changed': len(diff_files),
            'risk_level': 'low',
            'findings': [],
            'risky_files': []
        }
        
        max_risk = 'low'
        risk_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        for file_diff in diff_files:
            file_findings = []
            file_path = file_diff['new_path']
            
            # Skip non-code files
            if not any(file_path.endswith(ext) for ext in ['.cs', '.vb', '.cpp', '.h', '.java', '.py', '.js', '.ts']):
                continue
            
            # Check each chunk for patterns
            for chunk in file_diff['chunks']:
                chunk_text = '\n'.join(chunk['changes'])
                
                for category, pattern_info in self.bug_patterns.items():
                    for pattern in pattern_info['patterns']:
                        matches = re.findall(pattern, chunk_text, re.MULTILINE | re.IGNORECASE)
                        if matches:
                            finding = {
                                'category': category,
                                'description': pattern_info['description'],
                                'risk': pattern_info['risk'],
                                'file': file_path,
                                'line': chunk['new_start'],
                                'pattern_matches': len(matches),
                                'context': chunk['context']
                            }
                            file_findings.append(finding)
                            
                            # Update max risk
                            if risk_levels.get(pattern_info['risk'], 0) > risk_levels.get(max_risk, 0):
                                max_risk = pattern_info['risk']
            
            if file_findings:
                findings['findings'].extend(file_findings)
                findings['risky_files'].append({
                    'file': file_path,
                    'findings_count': len(file_findings),
                    'additions': file_diff['additions'],
                    'deletions': file_diff['deletions']
                })
        
        findings['risk_level'] = max_risk
        
        # Additional analysis
        findings['complexity_indicators'] = self.analyze_complexity(diff_files)
        
        return findings

    def analyze_complexity(self, diff_files: List[Dict]) -> Dict:
        """Analyze complexity indicators"""
        indicators = {
            'large_files_modified': 0,
            'high_churn_files': [],
            'cross_cutting_changes': False,
            'test_coverage': 'unknown'
        }
        
        # Check for large modifications
        for file_diff in diff_files:
            total_changes = file_diff['additions'] + file_diff['deletions']
            if total_changes > 100:
                indicators['large_files_modified'] += 1
                indicators['high_churn_files'].append({
                    'file': file_diff['new_path'],
                    'changes': total_changes
                })
        
        # Check for cross-cutting concerns
        unique_dirs = set()
        has_tests = False
        for file_diff in diff_files:
            path_parts = file_diff['new_path'].split('/')
            if len(path_parts) > 1:
                unique_dirs.add(path_parts[0])
            
            # Check for test files
            if any(test_indicator in file_diff['new_path'].lower() 
                   for test_indicator in ['test', 'spec', 'tests']):
                has_tests = True
        
        indicators['cross_cutting_changes'] = len(unique_dirs) > 3
        indicators['test_coverage'] = 'included' if has_tests else 'missing'
        
        return indicators

    def analyze_all_prs(self):
        """Analyze all PRs in the repository"""
        print("Starting PR diff analysis...")
        
        # Get all PR files
        pr_files = [f for f in os.listdir(self.prs_dir) if f.startswith('pr-') and f.endswith('.json')]
        self.analysis_results['total_prs'] = len(pr_files)
        
        for pr_file in pr_files:
            pr_number = int(pr_file.replace('pr-', '').replace('.json', ''))
            
            # Load PR data
            pr_data = self.load_pr_data(pr_number)
            if not pr_data:
                continue
            
            # Load diff
            diff_content = self.load_diff(pr_number)
            if not diff_content:
                continue
            
            # Parse and analyze
            diff_files = self.parse_diff(diff_content)
            findings = self.analyze_diff_for_bugs(diff_files, pr_data)
            
            self.analysis_results['analyzed_prs'] += 1
            
            # Track results
            if findings['risk_level'] in ['high', 'critical']:
                self.analysis_results['risky_prs'].append(findings)
            
            self.analysis_results['risk_summary'][findings['risk_level']] += 1
            
            # Track large PRs
            if findings['total_changes'] > 500:
                self.analysis_results['large_prs'].append({
                    'pr_number': pr_number,
                    'title': pr_data.get('title', 'Unknown'),
                    'changes': findings['total_changes'],
                    'files': findings['files_changed']
                })
            
            # Track findings by category
            for finding in findings['findings']:
                self.analysis_results['bug_patterns_found'][finding['category']].append({
                    'pr_number': pr_number,
                    'file': finding['file'],
                    'risk': finding['risk']
                })
            
            # Track files with most changes
            for file_info in findings['risky_files']:
                self.analysis_results['files_with_most_changes'][file_info['file']] += file_info['findings_count']
            
            # Progress indicator
            if self.analysis_results['analyzed_prs'] % 10 == 0:
                print(f"Analyzed {self.analysis_results['analyzed_prs']} PRs...")

    def generate_report(self, output_file: str = 'pr_diff_analysis_report.md'):
        """Generate comprehensive analysis report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# PR Diff Analysis Report - Potential Bug Detection\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- Total PRs in repository: {self.analysis_results['total_prs']}\n")
            f.write(f"- PRs analyzed: {self.analysis_results['analyzed_prs']}\n")
            f.write(f"- High-risk PRs identified: {len(self.analysis_results['risky_prs'])}\n")
            f.write(f"- Total bug patterns detected: {sum(len(v) for v in self.analysis_results['bug_patterns_found'].values())}\n\n")
            
            # Risk Distribution
            f.write("## Risk Level Distribution\n\n")
            f.write("| Risk Level | Count | Percentage |\n")
            f.write("|------------|-------|------------|\n")
            total_analyzed = self.analysis_results['analyzed_prs']
            for risk in ['critical', 'high', 'medium', 'low']:
                count = self.analysis_results['risk_summary'].get(risk, 0)
                percentage = (count / total_analyzed * 100) if total_analyzed > 0 else 0
                f.write(f"| {risk.capitalize()} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
            
            # High-Risk PRs
            f.write("## High-Risk Pull Requests\n\n")
            if self.analysis_results['risky_prs']:
                # Sort by risk level and number of findings
                sorted_risky = sorted(
                    self.analysis_results['risky_prs'],
                    key=lambda x: (x['risk_level'] == 'critical', len(x['findings'])),
                    reverse=True
                )
                
                for pr in sorted_risky[:20]:  # Top 20
                    f.write(f"### PR #{pr['pr_number']}: {pr['pr_title']}\n\n")
                    f.write(f"- **Author**: {pr['author']}\n")
                    f.write(f"- **Created**: {pr['created_at'][:10]}\n")
                    f.write(f"- **Status**: {'Merged' if pr['merged'] else 'Open'}\n")
                    f.write(f"- **Risk Level**: {pr['risk_level'].upper()}\n")
                    f.write(f"- **Total Changes**: {pr['total_changes']} lines\n")
                    f.write(f"- **Files Changed**: {pr['files_changed']}\n\n")
                    
                    # Group findings by category
                    findings_by_category = defaultdict(list)
                    for finding in pr['findings']:
                        findings_by_category[finding['category']].append(finding)
                    
                    f.write("**Findings:**\n\n")
                    for category, findings in findings_by_category.items():
                        f.write(f"- **{category.replace('_', ' ').title()}** ({len(findings)} issues)\n")
                        for finding in findings[:3]:  # Show first 3
                            f.write(f"  - {finding['file']} (line ~{finding['line']})\n")
                    
                    # Complexity indicators
                    complexity = pr['complexity_indicators']
                    if complexity['large_files_modified'] > 0:
                        f.write(f"\n**Complexity Warning**: {complexity['large_files_modified']} large files modified\n")
                    if complexity['test_coverage'] == 'missing':
                        f.write("\n**⚠️ No test changes detected**\n")
                    
                    f.write("\n---\n\n")
            else:
                f.write("No high-risk PRs identified.\n\n")
            
            # Bug Pattern Analysis
            f.write("## Bug Pattern Analysis\n\n")
            pattern_summary = []
            for category, occurrences in self.analysis_results['bug_patterns_found'].items():
                if occurrences:
                    pattern_summary.append({
                        'category': category,
                        'count': len(occurrences),
                        'high_risk_count': len([o for o in occurrences if o['risk'] in ['high', 'critical']])
                    })
            
            pattern_summary.sort(key=lambda x: x['count'], reverse=True)
            
            f.write("| Pattern Category | Total Occurrences | High Risk |\n")
            f.write("|------------------|-------------------|------------|\n")
            for pattern in pattern_summary:
                f.write(f"| {pattern['category'].replace('_', ' ').title()} | {pattern['count']} | {pattern['high_risk_count']} |\n")
            f.write("\n")
            
            # Most Modified Files
            f.write("## Files with Most Risk Indicators\n\n")
            top_files = sorted(
                self.analysis_results['files_with_most_changes'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            for file_path, count in top_files:
                f.write(f"- `{file_path}`: {count} risk indicators\n")
            f.write("\n")
            
            # Large PRs
            f.write("## Large Pull Requests (>500 changes)\n\n")
            if self.analysis_results['large_prs']:
                f.write("Large PRs are more likely to introduce bugs and are harder to review.\n\n")
                sorted_large = sorted(
                    self.analysis_results['large_prs'],
                    key=lambda x: x['changes'],
                    reverse=True
                )
                
                for pr in sorted_large[:10]:
                    f.write(f"- PR #{pr['pr_number']}: {pr['title']}\n")
                    f.write(f"  - Changes: {pr['changes']} lines across {pr['files']} files\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, here are key recommendations:\n\n")
            
            critical_count = self.analysis_results['risk_summary'].get('critical', 0)
            high_count = self.analysis_results['risk_summary'].get('high', 0)
            
            if critical_count > 0:
                f.write(f"1. **🚨 URGENT**: Review {critical_count} critical risk PRs immediately\n")
            
            if high_count > 0:
                f.write(f"2. **⚠️ HIGH PRIORITY**: {high_count} high-risk PRs require thorough review\n")
            
            # Pattern-specific recommendations
            for category, occurrences in self.analysis_results['bug_patterns_found'].items():
                if len(occurrences) > 5:
                    if category == 'null_checks':
                        f.write("3. **Null Safety**: Multiple null check removals detected - review for NullReferenceException risks\n")
                    elif category == 'exception_handling':
                        f.write("4. **Exception Handling**: Significant changes to error handling - ensure proper error recovery\n")
                    elif category == 'concurrency':
                        f.write("5. **Thread Safety**: Concurrency changes detected - review for race conditions\n")
                    elif category == 'security':
                        f.write("6. **Security Review**: Security-sensitive changes detected - requires security team review\n")
            
            if len(self.analysis_results['large_prs']) > 5:
                f.write(f"\n7. **PR Size**: {len(self.analysis_results['large_prs'])} large PRs detected - consider splitting large changes\n")
            
            f.write("\n---\nEnd of Report")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("PR Diff Analysis for Bug Detection")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Analyze all PRs
        self.analyze_all_prs()
        
        # Generate report
        print("\nGenerating report...")
        self.generate_report()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print(f"Analysis complete in {elapsed:.2f} seconds")
        print(f"Analyzed {self.analysis_results['analyzed_prs']} PRs")
        print(f"Found {len(self.analysis_results['risky_prs'])} high-risk PRs")
        print("Report saved to: pr_diff_analysis_report.md")
        print("=" * 60)


if __name__ == "__main__":
    # Run the analysis
    data_dir = r"c:\Users\ykovalova\msbuild\data"
    analyzer = PRDiffAnalyzer(data_dir)
    analyzer.run_analysis()
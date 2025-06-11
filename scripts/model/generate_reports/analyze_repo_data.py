import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from typing import Dict, List, Set, Tuple, Any
import multiprocessing as mp
from functools import partial
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

class OptimizedRepoAnalyzer:
    def __init__(self, data_dir: str, cache_dir: str = None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.issues = []
        self.open_issues = []  # New: track only open issues
        self.prs = []
        self.diffs = {}
        self.duplicates = []
        self.overlaps = []
        self.categories = defaultdict(list)
        self.potentially_fixed_issues = []  # New: track potentially fixed issues
        
        # Precompiled regex patterns
        self.diff_pattern = re.compile(r'^diff --git a/(.*) b/(.*)$', re.MULTILINE)
        self.issue_ref_pattern = re.compile(r'(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s*#(\d+)', re.IGNORECASE)
        self.issue_mention_pattern = re.compile(r'#(\d+)')

    def load_data(self, use_cache=True):
        """Load all JSON data from the repository with caching"""
        cache_file = os.path.join(self.cache_dir, 'loaded_data.pkl')

        if use_cache and os.path.exists(cache_file):
            print("Loading from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.issues = cached_data['issues']
                    self.prs = cached_data['prs']
                    self.diffs = cached_data['diffs']
                    # Filter open issues
                    self.open_issues = [issue for issue in self.issues if issue and issue.get('state') == 'OPEN']
                    print(f"Loaded {len(self.issues)} total issues ({len(self.open_issues)} open), {len(self.prs)} PRs from cache")
                    return
            except Exception as e:
                print(f"Cache loading failed: {e}, loading from files...")

        # Load issues in parallel
        issues_dir = os.path.join(self.data_dir, 'raw-data', 'msbuild', 'backlog')
        if os.path.exists(issues_dir):
            issue_files = [os.path.join(issues_dir, f) for f in os.listdir(issues_dir) if f.endswith('.json')]
            with mp.Pool() as pool:
                self.issues = pool.map(self._load_json_file, issue_files)
        
        # Filter open issues
        self.open_issues = [issue for issue in self.issues if issue and issue.get('state') == 'OPEN']

        # Load PRs in parallel
        prs_dir = os.path.join(self.data_dir, 'raw-data', 'msbuild', 'prs')
        if os.path.exists(prs_dir):
            pr_files = [os.path.join(prs_dir, f) for f in os.listdir(prs_dir) if f.endswith('.json')]
            with mp.Pool() as pool:
                self.prs = pool.map(self._load_json_file, pr_files)
        
        # Load diffs (simplified - these are usually smaller)
        diffs_dir = os.path.join(self.data_dir, 'raw-data', 'msbuild', 'diffs')
        if os.path.exists(diffs_dir):
            for file in os.listdir(diffs_dir):
                if file.endswith('.diff'):
                    # Only store the filename, load content on demand
                    self.diffs[file] = os.path.join(diffs_dir, file)

        # Save to cache
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'issues': self.issues,
                        'prs': self.prs,
                        'diffs': self.diffs
                    }, f)
                print("Data saved to cache")
            except Exception as e:
                print(f"Failed to save cache: {e}")
    
    @staticmethod
    def _load_json_file(filepath):
        """Load a single JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def extract_text_features(self, item: Dict) -> str:
        """Extract text features from an issue or PR - optimized version"""
        if item is None:
            return ""
        
        # Use list for better performance than string concatenation
        text_parts = []

        # Add title and body
        text_parts.append(item.get('title', ''))
        text_parts.append(item.get('body', ''))

        # Add comments if available (limit to first 5 for performance)
        if 'comments' in item and 'nodes' in item['comments']:
            comments = item['comments']['nodes'][:5]  # Limit comments
            for comment in comments:
                if comment and 'body' in comment:
                    text_parts.append(comment['body'])
        
        return ' '.join(filter(None, text_parts)).lower()
    
    def find_duplicates_among_open_issues(self, threshold: float = 0.7):
        """Find duplicates only among open issues"""
        print("Finding duplicates among open issues...")
        all_texts = []
        
        # Only process open issues
        for issue in self.open_issues:
            all_texts.append(self.extract_text_features(issue))
        
        if len(all_texts) < 2:
            print("Not enough open issues to find duplicates")
            return

        print(f"Vectorizing {len(all_texts)} open issues...")
        # Use TF-IDF with optimized parameters
        vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features for performance
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2  # Ignore very rare terms
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        except Exception as e:
            print(f"Vectorization failed: {e}")
            return

        print("Computing similarities...")
        similarities = cosine_similarity(tfidf_matrix)

        # Find high similarity pairs
        for i in range(len(self.open_issues)):
            for j in range(i + 1, len(self.open_issues)):
                if similarities[i, j] > threshold:
                    item1 = self.open_issues[i]
                    item2 = self.open_issues[j]

                    self.duplicates.append({
                        'type1': 'issue',
                        'number1': item1['number'],
                        'title1': item1.get('title', 'No title'),
                        'type2': 'issue',
                        'number2': item2['number'],
                        'title2': item2.get('title', 'No title'),
                        'similarity': float(similarities[i, j])
                    })

        print(f"Found {len(self.duplicates)} potential duplicates among open issues")
    
    def find_potentially_fixed_issues(self):
        """Find open issues that might have been fixed by PRs"""
        print("Finding potentially fixed issues...")
        
        # Build a map of issue numbers to issue data for quick lookup
        issue_map = {issue['number']: issue for issue in self.open_issues}

        # Check each PR for issue references
        for pr in self.prs:
            if not pr:
                continue

            pr_text = self.extract_text_features(pr)
            pr_title = pr.get('title', '')
            
            # Find issue references in PR text
            fixed_issues = set()

            # Look for "fixes #123" pattern
            fix_matches = self.issue_ref_pattern.findall(pr_text)
            fixed_issues.update(int(num) for num in fix_matches)

            # Also check title separately (case sensitive)
            fix_matches_title = self.issue_ref_pattern.findall(pr_title)
            fixed_issues.update(int(num) for num in fix_matches_title)

            # Look for any issue mentions
            all_mentions = self.issue_mention_pattern.findall(pr_text)
            mentioned_issues = set(int(num) for num in all_mentions)

            # Check if any of these issues are still open
            for issue_num in fixed_issues:
                if issue_num in issue_map:
                    self.potentially_fixed_issues.append({
                        'issue_number': issue_num,
                        'issue_title': issue_map[issue_num].get('title', 'No title'),
                        'pr_number': pr['number'],
                        'pr_title': pr.get('title', 'No title'),
                        'pr_state': pr.get('state', 'UNKNOWN'),
                        'confidence': 'high',  # Explicitly mentions fixing
                        'pr_merged': pr.get('mergedAt') is not None
                    })

            # For mentioned but not explicitly fixed issues, add with lower confidence
            for issue_num in mentioned_issues - fixed_issues:
                if issue_num in issue_map:
                    # Check if PR title/body suggests it's a fix
                    if any(word in pr_text for word in ['fix', 'solve', 'resolve', 'address']):
                        self.potentially_fixed_issues.append({
                            'issue_number': issue_num,
                            'issue_title': issue_map[issue_num].get('title', 'No title'),
                            'pr_number': pr['number'],
                            'pr_title': pr.get('title', 'No title'),
                            'pr_state': pr.get('state', 'UNKNOWN'),
                            'confidence': 'medium',  # Mentioned with fix-related keywords
                            'pr_merged': pr.get('mergedAt') is not None
                        })
        
        # Also check for semantic similarity between open issues and merged PRs
        self._find_semantically_fixed_issues()

        print(f"Found {len(self.potentially_fixed_issues)} potentially fixed issues")
    
    def _find_semantically_fixed_issues(self):
        """Find issues that might be fixed based on semantic similarity with PRs"""
        # Get merged PRs only
        merged_prs = [pr for pr in self.prs if pr and pr.get('mergedAt') is not None]

        if not merged_prs or not self.open_issues:
            return

        # Extract texts
        issue_texts = [self.extract_text_features(issue) for issue in self.open_issues]
        pr_texts = [self.extract_text_features(pr) for pr in merged_prs]

        all_texts = issue_texts + pr_texts

        # Vectorize
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Split back into issues and PRs
            issue_vectors = tfidf_matrix[:len(issue_texts)]
            pr_vectors = tfidf_matrix[len(issue_texts):]

            # Compute similarities
            similarities = cosine_similarity(issue_vectors, pr_vectors)

            # Find high similarity pairs (threshold higher for semantic matching)
            for i, issue in enumerate(self.open_issues):
                for j, pr in enumerate(merged_prs):
                    if similarities[i, j] > 0.8:  # Higher threshold for semantic matching
                        # Check if not already found
                        already_found = any(
                            pf['issue_number'] == issue['number'] and pf['pr_number'] == pr['number']
                            for pf in self.potentially_fixed_issues
                        )

                        if not already_found:
                            self.potentially_fixed_issues.append({
                                'issue_number': issue['number'],
                                'issue_title': issue.get('title', 'No title'),
                                'pr_number': pr['number'],
                                'pr_title': pr.get('title', 'No title'),
                                'pr_state': pr.get('state', 'CLOSED'),
                                'confidence': 'low',  # Semantic similarity only
                                'pr_merged': True,
                                'similarity_score': float(similarities[i, j])
                            })
        except Exception as e:
            print(f"Semantic similarity analysis failed: {e}")

    def categorize_open_issues(self):
        """Categorize only open issues"""
        categories_keywords = {
            'bug': ['bug', 'error', 'fix', 'issue', 'problem', 'crash', 'fail', 'broken', 'exception'],
            'feature': ['feature', 'enhancement', 'add', 'new', 'implement', 'request'],
            'performance': ['performance', 'slow', 'speed', 'optimize', 'parallel', 'faster', 'efficiency'],
            'documentation': ['doc', 'documentation', 'readme', 'wiki', 'comment', 'example'],
            'test': ['test', 'unit test', 'testing', 'coverage', 'ci', 'validation'],
            'refactor': ['refactor', 'cleanup', 'reorganize', 'simplify', 'improve code'],
            'build': ['build', 'compile', 'msbuild', 'project', 'compilation'],
            'dependency': ['dependency', 'package', 'nuget', 'reference', 'version'],
            'breaking-change': ['breaking', 'backward', 'compatibility', 'migration'],
            'security': ['security', 'vulnerability', 'cve', 'exploit', 'secure']
        }
        
        # Compile regex patterns for better performance
        category_patterns = {}
        for category, keywords in categories_keywords.items():
            pattern = '|'.join(re.escape(keyword) for keyword in keywords)
            category_patterns[category] = re.compile(pattern, re.IGNORECASE)

        print("Categorizing open issues...")

        # Process only open issues
        for issue in self.open_issues:
            text = self.extract_text_features(issue)
            item_categories = set()
            
            # Check patterns
            for category, pattern in category_patterns.items():
                if pattern.search(text):
                    item_categories.add(category)

            # Check labels
            if 'labels' in issue:
                labels = issue['labels']
                if isinstance(labels, list):
                    for label in labels:
                        label_text = label if isinstance(label, str) else label.get('name', '')
                        for category, pattern in category_patterns.items():
                            if pattern.search(label_text):
                                item_categories.add(category)

            # Add to categories
            for category in item_categories:
                self.categories[category].append({
                    'type': 'issue',
                    'number': issue['number'],
                    'title': issue.get('title', 'No title'),
                    'state': 'OPEN',
                    'created': issue.get('createdAt', 'Unknown'),
                    'updated': issue.get('updatedAt', 'Unknown')
                })

            # If no category found, add to 'uncategorized'
            if not item_categories:
                self.categories['uncategorized'].append({
                    'type': 'issue',
                    'number': issue['number'],
                    'title': issue.get('title', 'No title'),
                    'state': 'OPEN',
                    'created': issue.get('createdAt', 'Unknown'),
                    'updated': issue.get('updatedAt', 'Unknown')
                })

        print(f"Categorized {len(self.open_issues)} open issues into {len(self.categories)} categories")
    
    def find_stale_open_issues(self, days: int = 180):
        """Find open issues that haven't been updated in a while"""
        stale = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for issue in self.open_issues:
            if 'updatedAt' not in issue:
                continue

            try:
                updated = datetime.fromisoformat(issue['updatedAt'].replace('Z', '+00:00'))
                if updated.replace(tzinfo=None) < cutoff_date:
                    stale.append({
                        'number': issue['number'],
                        'title': issue.get('title', 'No title'),
                        'last_updated': issue['updatedAt'],
                        'created': issue.get('createdAt', 'Unknown'),
                        'days_old': (datetime.now() - updated.replace(tzinfo=None)).days
                    })
            except Exception:
                continue
        
        return sorted(stale, key=lambda x: x['days_old'], reverse=True)
    
# ...existing code...

    def generate_report(self, output_file: str = 'analysis_report.md'):
        """Generate a comprehensive markdown report without truncation"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Open Issues Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total Issues: {len(self.issues)}\n")
            f.write(f"- **Open Issues: {len(self.open_issues)}**\n")
            f.write(f"- Closed Issues: {len(self.issues) - len(self.open_issues)}\n")
            f.write(f"- Total PRs: {len(self.prs)}\n")
            f.write(f"- Merged PRs: {len([pr for pr in self.prs if pr and pr.get('mergedAt')])}\n\n")
            
            # Potentially Fixed Issues
            f.write("## Potentially Fixed Open Issues\n\n")
            if self.potentially_fixed_issues:
                # Group by confidence
                high_conf = [pf for pf in self.potentially_fixed_issues if pf['confidence'] == 'high']
                med_conf = [pf for pf in self.potentially_fixed_issues if pf['confidence'] == 'medium']
                low_conf = [pf for pf in self.potentially_fixed_issues if pf['confidence'] == 'low']

                if high_conf:
                    f.write(f"### High Confidence (Explicitly mentioned as fixed) - {len(high_conf)} issues\n\n")
                    for pf in high_conf:
                        f.write(f"- **Issue #{pf['issue_number']}**: {pf['issue_title']}\n")
                        f.write(f"  - Fixed by PR #{pf['pr_number']}: {pf['pr_title']}")
                        f.write(f" (Merged: {'Yes' if pf['pr_merged'] else 'No'})\n\n")

                if med_conf:
                    f.write(f"### Medium Confidence (Mentioned with fix-related keywords) - {len(med_conf)} issues\n\n")
                    for pf in med_conf:
                        f.write(f"- **Issue #{pf['issue_number']}**: {pf['issue_title']}\n")
                        f.write(f"  - Possibly fixed by PR #{pf['pr_number']}: {pf['pr_title']}")
                        f.write(f" (Merged: {'Yes' if pf['pr_merged'] else 'No'})\n\n")

                if low_conf:
                    f.write(f"### Low Confidence (High semantic similarity) - {len(low_conf)} issues\n\n")
                    for pf in sorted(low_conf, key=lambda x: x.get('similarity_score', 0), reverse=True):
                        f.write(f"- **Issue #{pf['issue_number']}**: {pf['issue_title']}\n")
                        f.write(f"  - Similar to PR #{pf['pr_number']}: {pf['pr_title']}")
                        if 'similarity_score' in pf:
                            f.write(f" (Similarity: {pf['similarity_score']:.2%})")
                        f.write("\n\n")
            else:
                f.write("No potentially fixed issues found.\n\n")

            # Statistical summary
            f.write("\n## Statistical Summary\n\n")
            f.write("### Issue Age Distribution\n\n")
            if self.open_issues:
                ages = []
                issues_with_ages = []
                for issue in self.open_issues:
                    if 'createdAt' in issue:
                        try:
                            created = datetime.fromisoformat(issue['createdAt'].replace('Z', '+00:00'))
                            age_days = (datetime.now() - created.replace(tzinfo=None)).days
                            ages.append(age_days)
                            issues_with_ages.append({
                                'issue': issue,
                                'age_days': age_days,
                                'created_date': created
                            })
                        except:
                            pass

                if ages:
                    ages.sort()
                    f.write(f"- Oldest issue: {max(ages)} days old\n")
                    f.write(f"- Newest issue: {min(ages)} days old\n")
                    f.write(f"- Median age: {ages[len(ages)//2]} days\n")
                    f.write(f"- Average age: {sum(ages)/len(ages):.0f} days\n\n")

                    # Add section for oldest issues
                    f.write("### Oldest Open Issues (Created 2+ Years Ago)\n\n")

                    # Sort by age (oldest first)
                    issues_with_ages.sort(key=lambda x: x['age_days'], reverse=True)

                    # Filter for issues older than 2 years (730 days)
                    old_issues = [item for item in issues_with_ages if item['age_days'] > 730]

                    if old_issues:
                        f.write(f"Total issues older than 2 years: {len(old_issues)}\n\n")

                        # Group by age ranges
                        extremely_old = [item for item in old_issues if item['age_days'] > 2920]  # 8+ years
                        very_old = [item for item in old_issues if 1825 <= item['age_days'] <= 2920]  # 5-8 years
                        old = [item for item in old_issues if 1095 <= item['age_days'] < 1825]  # 3-5 years
                        moderately_old = [item for item in old_issues if 730 <= item['age_days'] < 1095]  # 2-3 years

                        if extremely_old:
                            f.write(f"#### Extremely Old (8+ years) - {len(extremely_old)} issues\n\n")
                            for item in extremely_old[:10]:  # Show top 10
                                issue = item['issue']
                                f.write(f"- **Issue #{issue['number']}**: {issue.get('title', 'No title')}\n")
                                f.write(f"  - Created: {issue['createdAt'][:10]} ({item['age_days']} days ago, ~{item['age_days']//365} years)\n")
                                f.write(f"  - Last updated: {issue.get('updatedAt', 'Unknown')[:10]}\n")
                                if 'labels' in issue and issue['labels']:
                                    labels = [l.get('name', '') if isinstance(l, dict) else str(l) for l in issue['labels']]
                                    f.write(f"  - Labels: {', '.join(labels)}\n")
                                f.write("\n")

                        if very_old:
                            f.write(f"#### Very Old (5-8 years) - {len(very_old)} issues\n\n")
                            for item in very_old[:10]:  # Show top 10
                                issue = item['issue']
                                f.write(f"- **Issue #{issue['number']}**: {issue.get('title', 'No title')}\n")
                                f.write(f"  - Created: {issue['createdAt'][:10]} ({item['age_days']} days ago, ~{item['age_days']//365} years)\n")
                                f.write(f"  - Last updated: {issue.get('updatedAt', 'Unknown')[:10]}\n")
                                f.write("\n")

                        if old:
                            f.write(f"#### Old (3-5 years) - {len(old)} issues\n\n")
                            for item in old[:10]:  # Show top 10
                                issue = item['issue']
                                f.write(f"- **Issue #{issue['number']}**: {issue.get('title', 'No title')}\n")
                                f.write(f"  - Created: {issue['createdAt'][:10]} ({item['age_days']} days ago, ~{item['age_days']//365} years)\n")
                                f.write(f"  - Last updated: {issue.get('updatedAt', 'Unknown')[:10]}\n")
                                f.write("\n")

                        if moderately_old:
                            f.write(f"#### Moderately Old (2-3 years) - {len(moderately_old)} issues\n\n")
                            f.write(f"Showing first 10 of {len(moderately_old)} issues:\n\n")
                            for item in moderately_old[:10]:  # Show top 10
                                issue = item['issue']
                                f.write(f"- **Issue #{issue['number']}**: {issue.get('title', 'No title')}\n")
                                f.write(f"  - Created: {issue['createdAt'][:10]} ({item['age_days']} days ago, ~{item['age_days']//365} years)\n")
                                f.write("\n")
                    else:
                        f.write("No open issues older than 2 years found.\n\n")

            # Most active issues (by comment count)
            f.write("### Most Discussed Open Issues\n\n")

            # Duplicate Open Issues
            f.write("## Duplicate Open Issues\n\n")
            if self.duplicates:
                f.write(f"### Total duplicate pairs found: {len(self.duplicates)}\n\n")
                # Group duplicates by similarity ranges
                very_high = [d for d in self.duplicates if d['similarity'] >= 0.95]
                high = [d for d in self.duplicates if 0.90 <= d['similarity'] < 0.95]
                medium = [d for d in self.duplicates if 0.80 <= d['similarity'] < 0.90]
                low = [d for d in self.duplicates if 0.70 <= d['similarity'] < 0.80]

                if very_high:
                    f.write(f"#### Very High Similarity (95%+) - {len(very_high)} pairs\n\n")
                    for dup in sorted(very_high, key=lambda x: x['similarity'], reverse=True):
                        f.write(f"- **Issue #{dup['number1']}** vs **Issue #{dup['number2']}** "
                               f"(Similarity: {dup['similarity']:.2%})\n")
                        f.write(f"  - {dup['title1']}\n")
                        f.write(f"  - {dup['title2']}\n\n")

                if high:
                    f.write(f"#### High Similarity (90-95%) - {len(high)} pairs\n\n")
                    for dup in sorted(high, key=lambda x: x['similarity'], reverse=True):
                        f.write(f"- **Issue #{dup['number1']}** vs **Issue #{dup['number2']}** "
                               f"(Similarity: {dup['similarity']:.2%})\n")
                        f.write(f"  - {dup['title1']}\n")
                        f.write(f"  - {dup['title2']}\n\n")

                if medium:
                    f.write(f"#### Medium Similarity (80-90%) - {len(medium)} pairs\n\n")
                    for dup in sorted(medium, key=lambda x: x['similarity'], reverse=True):
                        f.write(f"- **Issue #{dup['number1']}** vs **Issue #{dup['number2']}** "
                               f"(Similarity: {dup['similarity']:.2%})\n")
                        f.write(f"  - {dup['title1']}\n")
                        f.write(f"  - {dup['title2']}\n\n")

                if low:
                    f.write(f"#### Lower Similarity (70-80%) - {len(low)} pairs\n\n")
                    for dup in sorted(low, key=lambda x: x['similarity'], reverse=True):
                        f.write(f"- **Issue #{dup['number1']}** vs **Issue #{dup['number2']}** "
                               f"(Similarity: {dup['similarity']:.2%})\n")
                        f.write(f"  - {dup['title1']}\n")
                        f.write(f"  - {dup['title2']}\n\n")
            else:
                f.write("No duplicates found among open issues.\n\n")
            
            # Categories
            f.write("## Open Issues by Category\n\n")
            f.write("### Category Summary\n\n")
            # Show summary table
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")
            total_categorized = sum(len(items) for items in self.categories.values())
            for category, items in sorted(self.categories.items(), key=lambda x: len(x[1]), reverse=True):
                percentage = (len(items) / len(self.open_issues)) * 100 if self.open_issues else 0
                f.write(f"| {category.capitalize()} | {len(items)} | {percentage:.1f}% |\n")
            f.write("\n")

            # Detailed category listings
            for category, items in sorted(self.categories.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"### {category.capitalize()} ({len(items)} issues)\n\n")
                # Sort by update date, most recent first
                sorted_items = sorted(items, key=lambda x: x.get('updated', ''), reverse=True)
                for item in sorted_items:
                    f.write(f"- Issue #{item['number']}: {item['title']}\n")
                    f.write(f"  - Last updated: {item.get('updated', 'Unknown')[:10]}\n")
                    f.write(f"  - Created: {item.get('created', 'Unknown')[:10]}\n")
                f.write("\n")
            
            # Stale issues
            stale_issues = self.find_stale_open_issues()
            f.write("## Stale Open Issues (not updated in 6+ months)\n\n")
            f.write(f"### Total stale issues: {len(stale_issues)}\n\n")

            if stale_issues:
                # Group by age
                very_old = [s for s in stale_issues if s['days_old'] > 730]  # 2+ years
                old = [s for s in stale_issues if 365 <= s['days_old'] <= 730]  # 1-2 years
                moderate = [s for s in stale_issues if 180 <= s['days_old'] < 365]  # 6m-1y

                if very_old:
                    f.write(f"#### Very Old (2+ years) - {len(very_old)} issues\n\n")
                    for issue in sorted(very_old, key=lambda x: x['days_old'], reverse=True):
                        f.write(f"- Issue #{issue['number']}: {issue['title']}\n")
                        f.write(f"  - Last updated: {issue['last_updated'][:10]} ({issue['days_old']} days ago)\n")
                        f.write(f"  - Created: {issue['created'][:10]}\n")
                    f.write("\n")

                if old:
                    f.write(f"#### Old (1-2 years) - {len(old)} issues\n\n")
                    for issue in sorted(old, key=lambda x: x['days_old'], reverse=True):
                        f.write(f"- Issue #{issue['number']}: {issue['title']}\n")
                        f.write(f"  - Last updated: {issue['last_updated'][:10]} ({issue['days_old']} days ago)\n")
                        f.write(f"  - Created: {issue['created'][:10]}\n")
                    f.write("\n")

                if moderate:
                    f.write(f"#### Moderately Old (6 months - 1 year) - {len(moderate)} issues\n\n")
                    for issue in sorted(moderate, key=lambda x: x['days_old'], reverse=True):
                        f.write(f"- Issue #{issue['number']}: {issue['title']}\n")
                        f.write(f"  - Last updated: {issue['last_updated'][:10]} ({issue['days_old']} days ago)\n")
                        f.write(f"  - Created: {issue['created'][:10]}\n")
                    f.write("\n")
            else:
                f.write("No stale issues found.\n")

            # Category overlap analysis
            f.write("\n## Category Overlap Analysis\n\n")
            multi_category_issues = defaultdict(list)
            for category, items in self.categories.items():
                for item in items:
                    multi_category_issues[item['number']].append(category)

            # Group by number of categories
            category_counts = defaultdict(list)
            for issue_num, categories in multi_category_issues.items():
                if len(categories) > 1:
                    category_counts[len(categories)].append((issue_num, categories))

            f.write(f"### Issues with multiple categories: {sum(len(issues) for issues in category_counts.values())}\n\n")

            for cat_count in sorted(category_counts.keys(), reverse=True):
                issues = category_counts[cat_count]
                f.write(f"#### Issues with {cat_count} categories ({len(issues)} issues)\n\n")

                for issue_num, categories in sorted(issues, key=lambda x: x[0]):
                    issue = next((i for i in self.open_issues if i['number'] == issue_num), None)
                    if issue:
                        f.write(f"- Issue #{issue_num}: {issue.get('title', 'No title')}\n")
                        f.write(f"  - Categories: {', '.join(sorted(categories))}\n")
                f.write("\n")

            # Statistical summary
            f.write("\n## Statistical Summary\n\n")
            f.write("### Issue Age Distribution\n\n")
            if self.open_issues:
                ages = []
                for issue in self.open_issues:
                    if 'createdAt' in issue:
                        try:
                            created = datetime.fromisoformat(issue['createdAt'].replace('Z', '+00:00'))
                            age_days = (datetime.now() - created.replace(tzinfo=None)).days
                            ages.append(age_days)
                        except:
                            pass

                if ages:
                    ages.sort()
                    f.write(f"- Oldest issue: {max(ages)} days old\n")
                    f.write(f"- Newest issue: {min(ages)} days old\n")
                    f.write(f"- Median age: {ages[len(ages)//2]} days\n")
                    f.write(f"- Average age: {sum(ages)/len(ages):.0f} days\n\n")

            # Most active issues (by comment count)
            f.write("### Most Discussed Open Issues\n\n")
            issues_with_comments = []
            for issue in self.open_issues:
                if 'comments' in issue and 'nodes' in issue['comments']:
                    comment_count = len(issue['comments']['nodes'])
                    if comment_count > 0:
                        issues_with_comments.append({
                            'number': issue['number'],
                            'title': issue.get('title', 'No title'),
                            'comments': comment_count
                        })

            issues_with_comments.sort(key=lambda x: x['comments'], reverse=True)
            for issue in issues_with_comments[:20]:
                f.write(f"- Issue #{issue['number']}: {issue['title']} ({issue['comments']} comments)\n")

            f.write("\n---\nEnd of Report")
    
    def analyze(self, use_cache=True):
        """Run the complete analysis with progress tracking"""
        start_time = datetime.now()
        
        print("=" * 50)
        print("Open Issues Analysis")
        print("=" * 50)
        
        # Load data
        print("\n[1/6] Loading data...")
        load_start = datetime.now()
        self.load_data(use_cache=use_cache)
        print(f"Data loading took: {(datetime.now() - load_start).total_seconds():.2f} seconds")
        
        # Find duplicates among open issues
        print("\n[2/6] Finding duplicates among open issues...")
        dup_start = datetime.now()
        self.find_duplicates_among_open_issues()
        print(f"Duplicate finding took: {(datetime.now() - dup_start).total_seconds():.2f} seconds")
        
        # Categorize open issues
        print("\n[3/6] Categorizing open issues...")
        cat_start = datetime.now()
        self.categorize_open_issues()
        print(f"Categorization took: {(datetime.now() - cat_start).total_seconds():.2f} seconds")

        # Find potentially fixed issues
        print("\n[4/6] Finding potentially fixed issues...")
        fix_start = datetime.now()
        self.find_potentially_fixed_issues()
        print(f"Fixed issue detection took: {(datetime.now() - fix_start).total_seconds():.2f} seconds")

        # Find stale issues
        print("\n[5/6] Analyzing stale issues...")
        stale_start = datetime.now()
        stale_issues = self.find_stale_open_issues()
        print(f"Found {len(stale_issues)} stale issues")
        print(f"Stale issue analysis took: {(datetime.now() - stale_start).total_seconds():.2f} seconds")

        # Generate report
        print("\n[6/6] Generating report...")
        report_start = datetime.now()
        self.generate_report()
        print(f"Report generation took: {(datetime.now() - report_start).total_seconds():.2f} seconds")
        
        # Summary
        total_time = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 50)
        print(f"Analysis complete in {total_time:.2f} seconds!")
        print(f"Check analysis_report.md for results.")
        print("=" * 50)

        # Save summary statistics
        stats = {
            'total_time': total_time,
            'total_issues': len(self.issues),
            'open_issues': len(self.open_issues),
            'closed_issues': len(self.issues) - len(self.open_issues),
            'prs_count': len(self.prs),
            'duplicates_found': len(self.duplicates),
            'potentially_fixed': len(self.potentially_fixed_issues),
            'stale_issues': len(stale_issues),
            'categories': {cat: len(items) for cat, items in self.categories.items()}
        }

        with open('analysis_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    # Set the data directory path
    data_dir = r"c:\Users\ykovalova\msbuild\data"
    
    # Create analyzer and run analysis
    analyzer = OptimizedRepoAnalyzer(data_dir)

    # Run with caching enabled (set to False to force reload)
    analyzer.analyze(use_cache=True)
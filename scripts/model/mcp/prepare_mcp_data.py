#!/usr/bin/env python3
"""
Prepare and convert MSBuild data for MCP server
Creates optimized SQLite database for fast queries
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict
import re

class MCPDataPreparer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "msbuild_analytics.db"
        self.conn = None
        
    def prepare(self):
        """Main preparation process"""
        print("MSBuild MCP Data Preparation")
        print("=" * 50)
        
        # Create database
        print("Creating database...")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create schema
        print("Creating schema...")
        self._create_schema()
        
        # Import data
        print("\nImporting data...")
        self._import_issues()
        self._import_pull_requests()
        self._import_diffs_metadata()
        
        # Create additional analytics tables
        print("\nCreating analytics tables...")
        self._create_analytics_tables()
        
        # Create views for common queries
        print("Creating optimized views...")
        self._create_views()
        
        print("\nOptimizing database...")
        self.conn.execute("VACUUM")
        self.conn.execute("ANALYZE")
        
        print("\nDatabase preparation complete!")
        self._print_summary()
    
    def _create_schema(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY,
                number INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                state TEXT NOT NULL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                closed_at TIMESTAMP,
                author TEXT,
                labels TEXT,
                comments_count INTEGER DEFAULT 0,
                milestone TEXT,
                assignees TEXT,
                url TEXT
            )
        """)
        
        # Pull requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pull_requests (
                id INTEGER PRIMARY KEY,
                number INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                state TEXT NOT NULL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                merged_at TIMESTAMP,
                closed_at TIMESTAMP,
                author TEXT,
                labels TEXT,
                commits_count INTEGER DEFAULT 0,
                additions INTEGER DEFAULT 0,
                deletions INTEGER DEFAULT 0,
                changed_files INTEGER DEFAULT 0,
                milestone TEXT,
                assignees TEXT,
                base_branch TEXT,
                head_branch TEXT,
                url TEXT
            )
        """)
        
        # Comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_number INTEGER,
                pr_number INTEGER,
                author TEXT,
                body TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (issue_number) REFERENCES issues(number),
                FOREIGN KEY (pr_number) REFERENCES pull_requests(number)
            )
        """)
        
        # Commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sha TEXT UNIQUE NOT NULL,
                pr_number INTEGER,
                author TEXT,
                message TEXT,
                created_at TIMESTAMP,
                additions INTEGER DEFAULT 0,
                deletions INTEGER DEFAULT 0,
                FOREIGN KEY (pr_number) REFERENCES pull_requests(number)
            )
        """)
        
        # File changes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_sha TEXT,
                filename TEXT,
                additions INTEGER DEFAULT 0,
                deletions INTEGER DEFAULT 0,
                change_type TEXT,
                FOREIGN KEY (commit_sha) REFERENCES commits(sha)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_state ON issues(state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_created ON issues(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_updated ON issues(updated_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_author ON issues(author)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prs_state ON pull_requests(state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prs_created ON pull_requests(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prs_merged ON pull_requests(merged_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prs_author ON pull_requests(author)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_issue ON comments(issue_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_pr ON comments(pr_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_created ON comments(created_at)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_pr ON commits(pr_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_commit ON file_changes(commit_sha)")
        
        # Full-text search tables
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS issues_fts USING fts5(
                number, title, body, content=issues, content_rowid=id
            )
        """)
        
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS prs_fts USING fts5(
                number, title, body, content=pull_requests, content_rowid=id
            )
        """)
        
        self.conn.commit()
    
    def _import_issues(self):
        """Import issues from JSON files"""
        # Check for the correct path structure
        issues_dir = self.data_dir / "raw-data" / "msbuild" / "backlog"
        if not issues_dir.exists():
            # Try alternative path
            issues_dir = self.data_dir / "msbuild" / "backlog"
            if not issues_dir.exists():
                print(f"  ERROR: No issues directory found at {issues_dir}")
                print(f"  Looking for: {self.data_dir}/raw-data/msbuild/backlog/")
                return
        
        cursor = self.conn.cursor()
        issue_count = 0
        comment_count = 0
        
        issue_files = list(issues_dir.glob("issue-*.json"))
        print(f"  Found {len(issue_files)} issue files")
        
        for i, issue_file in enumerate(issue_files):
            if i % 100 == 0:
                print(f"    Processing issue {i}/{len(issue_files)}...", end='\r')
            
            try:
                with open(issue_file, 'r', encoding='utf-8') as f:
                    issue = json.load(f)
                
                # Prepare data
                labels = json.dumps(issue.get('labels', []))
                assignees = json.dumps([a['login'] for a in issue.get('assignees', {}).get('nodes', [])])
                
                # Insert issue
                cursor.execute("""
                    INSERT OR REPLACE INTO issues 
                    (number, title, body, state, created_at, updated_at, closed_at,
                     author, labels, comments_count, milestone, assignees, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    issue.get('number'),
                    issue.get('title', ''),
                    issue.get('body', ''),
                    issue.get('state', 'UNKNOWN'),
                    issue.get('createdAt'),
                    issue.get('updatedAt'),
                    issue.get('closedAt'),
                    issue.get('author', {}).get('login') if issue.get('author') else None,
                    labels,
                    len(issue.get('comments', {}).get('nodes', [])),
                    issue.get('milestone', {}).get('title') if issue.get('milestone') else None,
                    assignees,
                    issue.get('url')
                ))
                
                # Insert into FTS
                cursor.execute("""
                    INSERT OR REPLACE INTO issues_fts (number, title, body)
                    VALUES (?, ?, ?)
                """, (
                    issue.get('number'),
                    issue.get('title', ''),
                    issue.get('body', '')
                ))
                
                # Import comments
                if 'comments' in issue and 'nodes' in issue['comments']:
                    for comment in issue['comments']['nodes']:
                        if comment and comment.get('body'):
                            cursor.execute("""
                                INSERT INTO comments 
                                (issue_number, author, body, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                issue['number'],
                                comment.get('author', {}).get('login') if comment.get('author') else None,
                                comment.get('body'),
                                comment.get('createdAt'),
                                comment.get('updatedAt')
                            ))
                            comment_count += 1
                
                issue_count += 1
                
                if issue_count % 100 == 0:
                    self.conn.commit()
            
            except Exception as e:
                print(f"\n  Error importing {issue_file}: {e}")
        
        self.conn.commit()
        print(f"\n  Imported {issue_count} issues with {comment_count} comments")
    
    def _import_pull_requests(self):
        """Import pull requests from JSON files"""
        # Check for the correct path structure
        prs_dir = self.data_dir / "raw-data" / "msbuild" / "prs"
        if not prs_dir.exists():
            # Try alternative path
            prs_dir = self.data_dir / "msbuild" / "prs"
            if not prs_dir.exists():
                print(f"  ERROR: No PRs directory found at {prs_dir}")
                print(f"  Looking for: {self.data_dir}/raw-data/msbuild/prs/")
                return
        
        cursor = self.conn.cursor()
        pr_count = 0
        commit_count = 0
        
        pr_files = list(prs_dir.glob("pr-*.json"))
        print(f"  Found {len(pr_files)} PR files")
        
        for i, pr_file in enumerate(pr_files):
            if i % 50 == 0:
                print(f"    Processing PR {i}/{len(pr_files)}...", end='\r')
            
            try:
                with open(pr_file, 'r', encoding='utf-8') as f:
                    pr = json.load(f)
                
                # Prepare data
                labels = json.dumps(pr.get('labels', []))
                assignees = json.dumps([a['login'] for a in pr.get('assignees', {}).get('nodes', [])])
                
                # Calculate stats
                additions = pr.get('additions', 0)
                deletions = pr.get('deletions', 0)
                changed_files = pr.get('changedFiles', 0)
                commits_count_val = len(pr.get('commits', {}).get('nodes', []))
                
                # Insert PR
                cursor.execute("""
                    INSERT OR REPLACE INTO pull_requests 
                    (number, title, body, state, created_at, updated_at, merged_at, closed_at,
                     author, labels, commits_count, additions, deletions, changed_files,
                     milestone, assignees, base_branch, head_branch, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pr.get('number'),
                    pr.get('title', ''),
                    pr.get('body', ''),
                    pr.get('state', 'UNKNOWN'),
                    pr.get('createdAt'),
                    pr.get('updatedAt'),
                    pr.get('mergedAt'),
                    pr.get('closedAt'),
                    pr.get('author', {}).get('login') if pr.get('author') else None,
                    labels,
                    commits_count_val,
                    additions,
                    deletions,
                    changed_files,
                    pr.get('milestone', {}).get('title') if pr.get('milestone') else None,
                    assignees,
                    pr.get('baseRefName'),
                    pr.get('headRefName'),
                    pr.get('url')
                ))
                
                # Insert into FTS
                cursor.execute("""
                    INSERT OR REPLACE INTO prs_fts (number, title, body)
                    VALUES (?, ?, ?)
                """, (
                    pr.get('number'),
                    pr.get('title', ''),
                    pr.get('body', '')
                ))
                
                # Import commits
                if 'commits' in pr and 'nodes' in pr['commits']:
                    for commit_node in pr['commits']['nodes']:
                        if commit_node and 'commit' in commit_node:
                            commit = commit_node['commit']
                            cursor.execute("""
                                INSERT OR IGNORE INTO commits 
                                (sha, pr_number, author, message, created_at, additions, deletions)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                commit.get('oid'),
                                pr['number'],
                                commit.get('author', {}).get('name') if commit.get('author') else None,
                                commit.get('message', ''),
                                commit.get('committedDate'),
                                commit.get('additions', 0),
                                commit.get('deletions', 0)
                            ))
                            commit_count += 1
                
                pr_count += 1
                
                if pr_count % 50 == 0:
                    self.conn.commit()
            
            except Exception as e:
                print(f"\n  Error importing {pr_file}: {e}")
        
        self.conn.commit()
        print(f"\n  Imported {pr_count} PRs with {commit_count} commits")
    
    def _import_diffs_metadata(self):
        """Import metadata from diff files"""
        # Check for the correct path structure
        diffs_dir = self.data_dir / "raw-data" / "msbuild" / "diffs"
        if not diffs_dir.exists():
            # Try alternative path
            diffs_dir = self.data_dir / "msbuild" / "diffs"
            if not diffs_dir.exists():
                print(f"  No diffs directory found at {diffs_dir}")
                print(f"  Looking for: {self.data_dir}/raw-data/msbuild/diffs/")
                return
        
        cursor = self.conn.cursor()
        file_count = 0
        
        diff_files = list(diffs_dir.glob("*.diff"))
        print(f"  Found {len(diff_files)} diff files")
        
        diff_pattern = re.compile(r'^diff --git a/(.*) b/(.*)$', re.MULTILINE)
        
        for i, diff_file in enumerate(diff_files[:100]):  # Limit to first 100 for performance
            if i % 10 == 0:
                print(f"    Processing diff {i}/100...", end='\r')
            
            try:
                commit_sha = diff_file.stem
                
                with open(diff_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(100000)  # Read first 100KB
                
                # Extract file changes
                matches = diff_pattern.findall(content)
                for match in matches:
                    filename = match[0]
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO file_changes 
                        (commit_sha, filename, change_type)
                        VALUES (?, ?, ?)
                    """, (commit_sha, filename, 'modified'))
                    file_count += 1
                
                if i % 10 == 0:
                    self.conn.commit()
            
            except Exception as e:
                print(f"\n  Error processing {diff_file}: {e}")
        
        self.conn.commit()
        print(f"\n  Imported {file_count} file change records")
    
    def _create_analytics_tables(self):
        """Create pre-computed analytics tables"""
        cursor = self.conn.cursor()
        
        # Daily statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats AS
            SELECT 
                DATE(created_at) as date,
                'issue' as type,
                COUNT(*) as created_count,
                SUM(CASE WHEN state = 'OPEN' THEN 1 ELSE 0 END) as open_count,
                SUM(CASE WHEN state = 'CLOSED' THEN 1 ELSE 0 END) as closed_count
            FROM issues
            WHERE created_at IS NOT NULL
            GROUP BY DATE(created_at)
            
            UNION ALL
            
            SELECT 
                DATE(created_at) as date,
                'pr' as type,
                COUNT(*) as created_count,
                SUM(CASE WHEN state = 'OPEN' THEN 1 ELSE 0 END) as open_count,
                SUM(CASE WHEN state = 'MERGED' THEN 1 ELSE 0 END) as closed_count
            FROM pull_requests
            WHERE created_at IS NOT NULL
            GROUP BY DATE(created_at)
        """)
        
        # Label statistics - with proper JSON handling
        cursor.execute("DROP TABLE IF EXISTS label_stats")
        cursor.execute("""
            CREATE TABLE label_stats (
                label_name TEXT,
                type TEXT,
                total_count INTEGER,
                open_count INTEGER
            )
        """)
        
        # Process issues labels
        cursor.execute("SELECT number, labels, state FROM issues WHERE labels IS NOT NULL")
        label_counts = defaultdict(lambda: {"total": 0, "open": 0})

        for row in cursor.fetchall():
            try:
                labels_data = json.loads(row[1]) if row[1] else []
                state = row[2]

                for label in labels_data:
                    # Handle both string labels and dict labels
                    if isinstance(label, str):
                        label_name = label
                    elif isinstance(label, dict) and 'name' in label:
                        label_name = label['name']
                    else:
                        continue

                    label_counts[label_name]["total"] += 1
                    if state == 'OPEN':
                        label_counts[label_name]["open"] += 1
            except json.JSONDecodeError:
                print(f"  Warning: Malformed JSON in issue {row[0]}, skipping labels")
                continue

        # Insert label stats
        for label_name, counts in label_counts.items():
            cursor.execute("""
                INSERT INTO label_stats (label_name, type, total_count, open_count)
                VALUES (?, ?, ?, ?)
            """, (label_name, 'issue', counts["total"], counts["open"]))

        # Process PR labels similarly
        cursor.execute("SELECT number, labels, state FROM pull_requests WHERE labels IS NOT NULL")
        pr_label_counts = defaultdict(lambda: {"total": 0, "open": 0})

        for row in cursor.fetchall():
            try:
                labels_data = json.loads(row[1]) if row[1] else []
                state = row[2]

                for label in labels_data:
                    if isinstance(label, str):
                        label_name = label
                    elif isinstance(label, dict) and 'name' in label:
                        label_name = label['name']
                    else:
                        continue

                    pr_label_counts[label_name]["total"] += 1
                    if state == 'OPEN':
                        pr_label_counts[label_name]["open"] += 1
            except json.JSONDecodeError:
                print(f"  Warning: Malformed JSON in PR {row[0]}, skipping labels")
                continue

        for label_name, counts in pr_label_counts.items():
            cursor.execute("""
                INSERT INTO label_stats (label_name, type, total_count, open_count)
                VALUES (?, ?, ?, ?)
            """, (label_name, 'pr', counts["total"], counts["open"]))

        self.conn.commit()
        print("  Created analytics tables")
    
    def _create_views(self):
        """Create optimized views for common queries"""
        cursor = self.conn.cursor()
        
        # Active issues view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS active_issues AS
            SELECT * FROM issues
            WHERE state = 'OPEN'
            AND updated_at > datetime('now', '-90 days')
            ORDER BY updated_at DESC
        """)
        
        # Recent activity view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS recent_activity AS
            SELECT 
                'issue' as type,
                number,
                title,
                state,
                created_at,
                updated_at,
                author
            FROM issues
            WHERE updated_at > datetime('now', '-7 days')
            
            UNION ALL
            
            SELECT 
                'pr' as type,
                number,
                title,
                state,
                created_at,
                updated_at,
                author
            FROM pull_requests
            WHERE updated_at > datetime('now', '-7 days')
            
            ORDER BY updated_at DESC
        """)
        
        self.conn.commit()
        print("  Created optimized views")
    
    def _print_summary(self):
        """Print summary statistics"""
        cursor = self.conn.cursor()
        
        print("\nDatabase Summary:")
        print("-" * 30)
        
        cursor.execute("SELECT COUNT(*) FROM issues")
        print(f"Total issues: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM pull_requests")
        print(f"Total PRs: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM comments")
        print(f"Total comments: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM commits")
        print(f"Total commits: {cursor.fetchone()[0]}")
        
        # Database size
        db_size = self.db_path.stat().st_size / (1024 * 1024)
        print(f"\nDatabase size: {db_size:.1f} MB")
        print(f"Database location: {self.db_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_mcp_data.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    preparer = MCPDataPreparer(data_dir)
    preparer.prepare()

if __name__ == "__main__":
    main()
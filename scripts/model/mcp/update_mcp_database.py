#!/usr/bin/env python3
"""
Update MSBuild MCP database with latest data
Supports incremental updates and full refreshes
"""

import json
import sqlite3
import requests
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import subprocess
import time

class MCPDatabaseUpdater:
    def __init__(self, data_dir: str, github_token: str = None):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "msbuild_analytics.db"
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.conn = None
        
    def update(self, mode='incremental', days_back=7):
        """Main update process"""
        print("MSBuild MCP Database Updater")
        print("=" * 50)
        
        if mode == 'incremental':
            self._incremental_update(days_back)
        elif mode == 'full':
            self._full_refresh()
        elif mode == 'smart':
            self._smart_update()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _incremental_update(self, days_back=7):
        """Update only recent changes"""
        print(f"Performing incremental update (last {days_back} days)...")
        
        # Step 1: Fetch new data from GitHub
        print("\n1. Fetching latest data from GitHub...")
        self._fetch_recent_data(days_back)
        
        # Step 2: Update the database
        print("\n2. Updating database...")
        self._update_database_incremental()
        
        # Step 3: Update analytics
        print("\n3. Refreshing analytics tables...")
        self._refresh_analytics()
        
        print("\nIncremental update complete!")
    
    def _full_refresh(self):
        """Complete refresh of all data"""
        print("Performing full refresh...")
        
        # Step 1: Fetch all data
        print("\n1. Fetching all data from GitHub...")
        self._fetch_all_data()
        
        # Step 2: Rebuild database
        print("\n2. Rebuilding database...")
        script_path = Path(__file__).parent / "prepare_mcp_data.py"
        subprocess.run([
            sys.executable, 
            str(script_path),
            str(self.data_dir)
        ])
        
        print("\nFull refresh complete!")
    
    def _smart_update(self):
        """Smart update based on last update time"""
        print("Performing smart update...")
        
        # Check last update time
        last_update = self._get_last_update_time()
        
        if last_update:
            days_since = (datetime.now() - last_update).days
            print(f"Last update was {days_since} days ago")
            
            if days_since > 30:
                print("More than 30 days - doing full refresh")
                self._full_refresh()
            else:
                print(f"Doing incremental update for last {days_since + 1} days")
                self._incremental_update(days_since + 1)
        else:
            print("No previous update found - doing full refresh")
            self._full_refresh()
    
    def _fetch_recent_data(self, days_back):
        """Fetch recent issues and PRs"""
        since = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Use the robust fetcher script
        fetch_script = self.data_dir.parent / "scripts" / "model" / "robust_fetch_msbuild.py"
        
        if fetch_script.exists():
            print(f"Using robust fetcher (since {since[:10]})...")
            cmd = [
                sys.executable,
                str(fetch_script),
                "--since", since,
                "--max-issues", "200",
                "--max-prs", "100"
            ]
            
            if not self.github_token:
                cmd.append("--no-ssl-verify")
            
            subprocess.run(cmd)
        else:
            # Fallback to direct API calls
            self._fetch_via_api(since)
    
    def _fetch_via_api(self, since):
        """Direct API fetch (fallback)"""
        if not self.github_token:
            print("WARNING: No GitHub token set. API rate limits will apply.")
        
        session = requests.Session()
        if self.github_token:
            session.headers['Authorization'] = f'token {self.github_token}'
        
        # Fetch recent issues
        print("Fetching recent issues...")
        issues_fetched = 0
        page = 1
        
        while True:
            response = session.get(
                'https://api.github.com/repos/dotnet/msbuild/issues',
                params={
                    'state': 'all',
                    'since': since,
                    'per_page': 100,
                    'page': page
                }
            )
            
            if response.status_code != 200:
                print(f"Error fetching issues: {response.status_code}")
                break
            
            issues = response.json()
            if not issues:
                break
            
            # Save issues
            backlog_dir = self.data_dir / "raw-data" / "msbuild" / "backlog"
            backlog_dir.mkdir(parents=True, exist_ok=True)
            
            for issue in issues:
                if 'pull_request' not in issue:  # Skip PRs
                    issue_file = backlog_dir / f"issue-{issue['number']}.json"
                    
                    # Convert to expected format
                    issue_data = {
                        'number': issue['number'],
                        'title': issue['title'],
                        'body': issue['body'],
                        'state': issue['state'].upper(),
                        'createdAt': issue['created_at'],
                        'updatedAt': issue['updated_at'],
                        'closedAt': issue['closed_at'],
                        'author': {'login': issue['user']['login']} if issue.get('user') else None,
                        'labels': [label['name'] for label in issue.get('labels', [])],
                        'comments': {'nodes': []}  # Would need separate API call
                    }
                    
                    with open(issue_file, 'w', encoding='utf-8') as f:
                        json.dump(issue_data, f, indent=2)
                    
                    issues_fetched += 1
            
            print(f"  Fetched {issues_fetched} issues so far...")
            page += 1
            time.sleep(0.5)  # Rate limiting
        
        print(f"Total issues fetched: {issues_fetched}")
    
    def _update_database_incremental(self):
        """Update only changed records in database"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Get list of updated files
        issues_dir = self.data_dir / "raw-data" / "msbuild" / "backlog"
        prs_dir = self.data_dir / "raw-data" / "msbuild" / "prs"
        
        updated_count = 0
        
        # Update issues
        if issues_dir.exists():
            for issue_file in issues_dir.glob("issue-*.json"):
                # Check if file was modified recently
                if issue_file.stat().st_mtime > time.time() - (7 * 24 * 60 * 60):
                    with open(issue_file, 'r', encoding='utf-8') as f:
                        issue = json.load(f)
                    
                    # Update or insert
                    cursor.execute("""
                        INSERT OR REPLACE INTO issues 
                        (number, title, body, state, created_at, updated_at, closed_at,
                         author, labels, comments_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        issue.get('number'),
                        issue.get('title'),
                        issue.get('body'),
                        issue.get('state'),
                        issue.get('createdAt'),
                        issue.get('updatedAt'),
                        issue.get('closedAt'),
                        issue.get('author', {}).get('login') if issue.get('author') else None,
                        json.dumps(issue.get('labels', [])),
                        len(issue.get('comments', {}).get('nodes', []))
                    ))
                    
                    updated_count += 1
        
        self.conn.commit()
        print(f"Updated {updated_count} records")
    
    def _refresh_analytics(self):
        """Refresh analytics tables and views"""
        if not self.conn:
            self.conn = sqlite3.connect(str(self.db_path))
        
        cursor = self.conn.cursor()
        
        # Update daily stats for recent days
        cursor.execute("""
            DELETE FROM daily_stats 
            WHERE date > date('now', '-7 days')
        """)
        
        cursor.execute("""
            INSERT INTO daily_stats
            SELECT 
                DATE(created_at) as date,
                'issue' as type,
                COUNT(*) as created_count,
                SUM(CASE WHEN state = 'OPEN' THEN 1 ELSE 0 END) as open_count,
                SUM(CASE WHEN state = 'CLOSED' THEN 1 ELSE 0 END) as closed_count
            FROM issues
            WHERE created_at > date('now', '-7 days')
            GROUP BY DATE(created_at)
        """)
        
        # Update label stats
        cursor.execute("DROP TABLE IF EXISTS label_stats")
        cursor.execute("""
            CREATE TABLE label_stats AS
            WITH all_labels AS (
                SELECT 
                    json_extract(value, '$') as label_name,
                    'issue' as type,
                    state
                FROM issues, json_each(labels)
                WHERE labels IS NOT NULL
            )
            SELECT 
                label_name,
                type,
                COUNT(*) as total_count,
                SUM(CASE WHEN state IN ('OPEN') THEN 1 ELSE 0 END) as open_count
            FROM all_labels
            GROUP BY label_name, type
        """)
        
        # Update FTS indexes
        cursor.execute("INSERT INTO issues_fts(issues_fts) VALUES('rebuild')")
        
        # Record update time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY,
                update_time TIMESTAMP,
                update_type TEXT,
                records_updated INTEGER
            )
        """)
        
        cursor.execute("""
            INSERT INTO update_log (update_time, update_type, records_updated)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), 'incremental', 0))
        
        self.conn.commit()
        print("Analytics refreshed")
    
    def _get_last_update_time(self):
        """Get the last update timestamp"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(update_time) FROM update_log
            """)
            
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            
            return None
        except:
            return None
    
    def _fetch_all_data(self):
        """Fetch all data using existing scripts"""
        fetch_script = self.data_dir.parent / "scripts" / "model" / "fetch_all_issues.py"
        
        if fetch_script.exists():
            subprocess.run([sys.executable, str(fetch_script)])
        else:
            print("Full fetch script not found. Using robust fetcher...")
            subprocess.run([
                sys.executable,
                "robust_fetch_msbuild.py",
                "--max-issues", "5000",
                "--max-prs", "1000"
            ])

def create_update_script():
    """Create a simple update script"""
    script_content = '''#!/usr/bin/env python3
"""Simple update script for MSBuild MCP database"""

import subprocess
import sys
from datetime import datetime

print(f"MSBuild MCP Database Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Run the updater in smart mode
subprocess.run([
    sys.executable, 
    "update_mcp_database.py",
    "c:\\\\Users\\\\ykovalova\\\\msbuild\\\\data",
    "--mode", "smart"
])
'''
    
    with open("update_msbuild_mcp.py", "w") as f:
        f.write(script_content)
    
    print("Created update_msbuild_mcp.py")

def create_windows_task_scheduler():
    """Create Windows Task Scheduler XML"""
    xml_content = '''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Update MSBuild MCP Database Daily</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2024-01-01T02:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>python</Command>
      <Arguments>C:\\Users\\ykovalova\\msbuild\\scripts\\model\\update_msbuild_mcp.py</Arguments>
      <WorkingDirectory>C:\\Users\\ykovalova\\msbuild\\scripts\\model</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''
    
    with open("msbuild_mcp_update_task.xml", "w", encoding='utf-16') as f:
        f.write(xml_content)
    
    print("Created msbuild_mcp_update_task.xml")
    print("\nTo install:")
    print('schtasks /create /tn "MSBuild MCP Update" /xml msbuild_mcp_update_task.xml')

def main():
    parser = argparse.ArgumentParser(description='Update MSBuild MCP Database')
    parser.add_argument('data_dir', help='Data directory path')
    parser.add_argument('--mode', choices=['incremental', 'full', 'smart'], 
                       default='smart', help='Update mode')
    parser.add_argument('--days', type=int, default=7, 
                       help='Days to look back for incremental update')
    parser.add_argument('--create-scripts', action='store_true',
                       help='Create helper scripts')
    
    args = parser.parse_args()
    
    if args.create_scripts:
        create_update_script()
        create_windows_task_scheduler()
        return
    
    updater = MCPDatabaseUpdater(args.data_dir)
    updater.update(mode=args.mode, days_back=args.days)

if __name__ == "__main__":
    main()
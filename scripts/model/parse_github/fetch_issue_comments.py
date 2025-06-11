#!/usr/bin/env python3
"""
Fetch comments for MSBuild issues
Updates existing issue files with comment data
"""
import json
import requests
import time
from pathlib import Path
import os
import sys
import urllib3
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CommentFetcher:
    def __init__(self, token=None, verify_ssl=True):
        self.token = token
        self.session = requests.Session()
        self.session.verify = verify_ssl
        
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MSBuild-Comment-Fetcher'
        })
        
        if token:
            self.session.headers['Authorization'] = f'token {token}'
        
        self.stats = {
            'issues_processed': 0,
            'comments_fetched': 0,
            'issues_with_comments': 0,
            'issues_without_comments': 0,
            'errors': 0
        }
    
    def fetch_comments_for_issue(self, owner, repo, issue_number):
        """Fetch all comments for a single issue"""
        comments = []
        page = 1
        
        while True:
            try:
                response = self.session.get(
                    f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments',
                    params={'per_page': 100, 'page': page},
                    timeout=30
                )
                
                if response.status_code == 404:
                    # Issue might have been deleted
                    return None
                
                if response.status_code != 200:
                    print(f"Error fetching comments for #{issue_number}: {response.status_code}")
                    return None
                
                page_comments = response.json()
                if not page_comments:
                    break
                
                # Format comments to match expected structure
                for comment in page_comments:
                    formatted_comment = {
                        'body': comment.get('body', ''),
                        'createdAt': comment.get('created_at'),
                        'updatedAt': comment.get('updated_at'),
                        'author': {
                            'login': comment['user']['login'] if comment.get('user') else '[deleted]'
                        }
                    }
                    comments.append(formatted_comment)
                
                page += 1
                
            except Exception as e:
                print(f"Error fetching comments for #{issue_number}: {e}")
                self.stats['errors'] += 1
                return None
        
        return comments
    
    def update_all_issues(self, data_dir, max_issues=None):
        """Update all issue files with comments"""
        backlog_dir = data_dir / 'backlog'
        issue_files = list(backlog_dir.glob('issue-*.json'))
        
        if max_issues:
            issue_files = issue_files[:max_issues]
        
        print(f"Found {len(issue_files)} issue files to process")
        print("Fetching comments for each issue...\n")
        
        # Check rate limit before starting
        self._check_rate_limit()
        
        for i, issue_file in enumerate(issue_files):
            # Load issue data
            with open(issue_file, 'r', encoding='utf-8') as f:
                issue_data = json.load(f)
            
            issue_number = issue_data.get('number')
            
            # Check if comments already exist and are populated
            existing_comments = issue_data.get('comments', {}).get('nodes', [])
            if existing_comments and len(existing_comments) > 0:
                # Skip if we already have comments
                print(f"[{i+1}/{len(issue_files)}] Issue #{issue_number}: Already has {len(existing_comments)} comments, skipping")
                continue
            
            print(f"[{i+1}/{len(issue_files)}] Issue #{issue_number}: ", end='', flush=True)
            
            # Fetch comments
            comments = self.fetch_comments_for_issue('dotnet', 'msbuild', issue_number)
            
            if comments is None:
                print("ERROR")
                continue
            
            # Update issue data
            if 'comments' not in issue_data:
                issue_data['comments'] = {}
            issue_data['comments']['nodes'] = comments
            
            # Save updated issue
            with open(issue_file, 'w', encoding='utf-8') as f:
                json.dump(issue_data, f, indent=2)
            
            # Update stats
            self.stats['issues_processed'] += 1
            self.stats['comments_fetched'] += len(comments)
            
            if comments:
                self.stats['issues_with_comments'] += 1
                print(f"{len(comments)} comments")
            else:
                self.stats['issues_without_comments'] += 1
                print("No comments")
            
            # Rate limiting
            time.sleep(0.5)
            
            # Check rate limit periodically
            if (i + 1) % 50 == 0:
                self._check_rate_limit()
    
    def _check_rate_limit(self):
        """Check and display rate limit"""
        try:
            response = self.session.get('https://api.github.com/rate_limit')
            if response.status_code == 200:
                data = response.json()
                remaining = data['rate']['remaining']
                reset_time = datetime.fromtimestamp(data['rate']['reset'])
                
                print(f"\n📊 Rate limit: {remaining} requests remaining (resets at {reset_time.strftime('%H:%M:%S')})")
                
                if remaining < 100:
                    print("⚠️  Low rate limit!")
                    
                    if remaining < 10:
                        wait_time = (reset_time - datetime.now()).total_seconds() + 60
                        print(f"⏸️  Pausing for {wait_time:.0f} seconds...")
                        time.sleep(wait_time)
                print("")
        except:
            pass
    
    def fetch_comments_for_specific_issues(self, data_dir, issue_numbers):
        """Fetch comments for specific issue numbers"""
        backlog_dir = data_dir / 'backlog'
        
        for issue_number in issue_numbers:
            issue_file = backlog_dir / f'issue-{issue_number}.json'
            
            if not issue_file.exists():
                print(f"Issue #{issue_number}: File not found")
                continue
            
            print(f"Issue #{issue_number}: ", end='', flush=True)
            
            # Load issue
            with open(issue_file, 'r', encoding='utf-8') as f:
                issue_data = json.load(f)
            
            # Fetch comments
            comments = self.fetch_comments_for_issue('dotnet', 'msbuild', issue_number)
            
            if comments is None:
                print("ERROR")
                continue
            
            # Update and save
            if 'comments' not in issue_data:
                issue_data['comments'] = {}
            issue_data['comments']['nodes'] = comments
            
            with open(issue_file, 'w', encoding='utf-8') as f:
                json.dump(issue_data, f, indent=2)
            
            print(f"{len(comments)} comments")
            time.sleep(0.5)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch comments for MSBuild issues')
    parser.add_argument('--max', type=int, help='Maximum number of issues to process')
    parser.add_argument('--issues', type=int, nargs='+', help='Specific issue numbers to fetch comments for')
    parser.add_argument('--no-ssl-verify', action='store_true', help='Disable SSL verification')
    args = parser.parse_args()
    
    # Check token
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("⚠️  Warning: No GITHUB_TOKEN found.")
        print("You'll be limited to 60 requests/hour.")
        print("\nTo set token in PowerShell:")
        print('  $env:GITHUB_TOKEN="your_token_here"')
        
        response = input("\nContinue without token? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / 'data' / 'raw-data' / 'msbuild'
    
    # Create fetcher
    fetcher = CommentFetcher(token, verify_ssl=not args.no_ssl_verify)
    
    print("MSBuild Issue Comment Fetcher")
    print("="*50)
    
    start_time = time.time()
    
    if args.issues:
        # Fetch comments for specific issues
        print(f"Fetching comments for {len(args.issues)} specific issues...")
        fetcher.fetch_comments_for_specific_issues(data_dir, args.issues)
    else:
        # Fetch comments for all issues
        fetcher.update_all_issues(data_dir, max_issues=args.max)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Issues processed: {fetcher.stats['issues_processed']}")
    print(f"Total comments fetched: {fetcher.stats['comments_fetched']}")
    print(f"Issues with comments: {fetcher.stats['issues_with_comments']}")
    print(f"Issues without comments: {fetcher.stats['issues_without_comments']}")
    print(f"Errors: {fetcher.stats['errors']}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    
    if fetcher.stats['issues_processed'] > 0:
        avg_comments = fetcher.stats['comments_fetched'] / fetcher.stats['issues_processed']
        print(f"Average comments per issue: {avg_comments:.1f}")

if __name__ == '__main__':
    main()
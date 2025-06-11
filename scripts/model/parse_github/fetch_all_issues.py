#!/usr/bin/env python3
"""
Fetch ALL issues from MSBuild repository
No limits - gets everything!
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

class CompleteFetcher:
    def __init__(self, token=None, verify_ssl=True):
        self.token = token
        self.session = requests.Session()
        self.session.verify = verify_ssl
        
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MSBuild-Complete-Fetcher'
        })
        
        if token:
            self.session.headers['Authorization'] = f'token {token}'
    
    def fetch_all_issues(self, owner, repo, output_dir, state='all'):
        """Fetch ALL issues without any limit"""
        backlog_dir = output_dir / 'backlog'
        backlog_dir.mkdir(exist_ok=True)
        
        # Check existing
        existing_issues = {int(f.stem.split('-')[1]) for f in backlog_dir.glob('issue-*.json')}
        print(f"Found {len(existing_issues)} existing issues in cache")
        
        total_fetched = 0
        total_skipped = 0
        page = 1
        consecutive_empty_pages = 0
        
        print(f"\nFetching ALL {state} issues from {owner}/{repo}...")
        print("This may take a while...\n")
        
        while True:
            print(f"Fetching page {page}...", end='', flush=True)
            
            try:
                response = self.session.get(
                    f'https://api.github.com/repos/{owner}/{repo}/issues',
                    params={
                        'state': state,
                        'per_page': 100,  # Maximum allowed
                        'page': page,
                        'sort': 'created',  # Oldest first to get all
                        'direction': 'asc'
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f" ERROR: {response.status_code}")
                    if response.status_code == 403:
                        print("Rate limit hit! Waiting...")
                        self._handle_rate_limit(response)
                        continue
                    break
                
                items = response.json()
                
                if not items:
                    consecutive_empty_pages += 1
                    print(" (empty)")
                    if consecutive_empty_pages >= 3:
                        print("\nReached end of issues.")
                        break
                    page += 1
                    continue
                
                consecutive_empty_pages = 0
                page_fetched = 0
                page_skipped = 0
                
                for item in items:
                    # Skip pull requests
                    if 'pull_request' in item:
                        continue
                    
                    issue_number = item['number']
                    
                    if issue_number in existing_issues:
                        page_skipped += 1
                        total_skipped += 1
                        continue
                    
                    # Save issue
                    issue_data = {
                        'number': issue_number,
                        'title': item['title'],
                        'body': item['body'] or '',
                        'state': item['state'].upper(),
                        'createdAt': item['created_at'],
                        'updatedAt': item['updated_at'],
                        'closedAt': item['closed_at'],
                        'author': {'login': item['user']['login']} if item.get('user') else None,
                        'labels': [label['name'] for label in item.get('labels', [])],
                        'assignees': {'nodes': [{'login': a['login']} for a in item.get('assignees', [])]},
                        'milestone': {'title': item['milestone']['title']} if item.get('milestone') else None,
                        'comments': {'nodes': []}  # Would need separate API call
                    }
                    
                    issue_file = backlog_dir / f'issue-{issue_number}.json'
                    with open(issue_file, 'w', encoding='utf-8') as f:
                        json.dump(issue_data, f, indent=2)
                    
                    page_fetched += 1
                    total_fetched += 1
                
                print(f" fetched {page_fetched}, skipped {page_skipped}")
                
                # Rate limiting
                time.sleep(0.5)
                
                # Check rate limit periodically
                if page % 10 == 0:
                    self._check_rate_limit()
                
                page += 1
                
            except Exception as e:
                print(f" ERROR: {e}")
                time.sleep(2)
                continue
        
        return total_fetched, total_skipped
    
    def _check_rate_limit(self):
        """Check and display rate limit"""
        try:
            response = self.session.get('https://api.github.com/rate_limit')
            if response.status_code == 200:
                data = response.json()
                remaining = data['rate']['remaining']
                reset_time = datetime.fromtimestamp(data['rate']['reset'])
                
                if remaining < 100:
                    print(f"\n⚠️  Low rate limit: {remaining} requests remaining")
                    print(f"    Resets at: {reset_time}")
                    
                    if remaining < 10:
                        wait_time = (reset_time - datetime.now()).total_seconds() + 60
                        print(f"    Waiting {wait_time:.0f} seconds...")
                        time.sleep(wait_time)
        except:
            pass
    
    def _handle_rate_limit(self, response):
        """Handle rate limit response"""
        reset_time = response.headers.get('X-RateLimit-Reset')
        if reset_time:
            reset_dt = datetime.fromtimestamp(int(reset_time))
            wait_time = (reset_dt - datetime.now()).total_seconds() + 60
            print(f"Rate limited. Waiting {wait_time:.0f} seconds until {reset_dt}...")
            time.sleep(wait_time)
        else:
            print("Rate limited. Waiting 60 seconds...")
            time.sleep(60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch ALL MSBuild issues')
    parser.add_argument('--state', choices=['all', 'open', 'closed'], default='all',
                       help='State of issues to fetch (default: all)')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Disable SSL verification')
    args = parser.parse_args()
    
    # Check token
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("ERROR: GitHub token is REQUIRED for fetching all issues!")
        print("\nTo set token in PowerShell:")
        print('  $env:GITHUB_TOKEN="your_token_here"')
        print("\nWithout a token, you're limited to 60 requests/hour")
        print("which is not enough to fetch all issues.")
        sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / 'data' / 'raw-data' / 'msbuild'
    
    # Create fetcher
    fetcher = CompleteFetcher(token, verify_ssl=not args.no_ssl_verify)
    
    print(f"Output directory: {output_dir}")
    print(f"Fetching ALL {args.state} issues...")
    print("="*50)
    
    start_time = time.time()
    fetched, skipped = fetcher.fetch_all_issues('dotnet', 'msbuild', output_dir, state=args.state)
    elapsed = time.time() - start_time
    
    # Final count
    total_issues = len(list((output_dir / 'backlog').glob('issue-*.json')))
    
    print("\n" + "="*50)
    print("COMPLETE!")
    print("="*50)
    print(f"New issues fetched: {fetched}")
    print(f"Issues skipped (already existed): {skipped}")
    print(f"Total issues in dataset: {total_issues}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    
    # Also save a summary
    summary = {
        'fetch_date': datetime.now().isoformat(),
        'state_fetched': args.state,
        'new_issues': fetched,
        'skipped_issues': skipped,
        'total_issues': total_issues,
        'elapsed_seconds': elapsed
    }
    
    with open(output_dir / 'fetch_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
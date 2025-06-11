#!/usr/bin/env python3
"""
Robust MSBuild Data Fetcher with SSL error handling
Uses REST API only for better reliability
"""
import json
import requests
import time
from pathlib import Path
from datetime import datetime
import os
import sys
import urllib3

# Disable SSL warnings if needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class RobustFetcher:
    def __init__(self, token=None, verify_ssl=True):
        self.token = token
        self.verify_ssl = verify_ssl
        self.session = requests.Session()

        # Set headers
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MSBuild-Dataset-Fetcher'
        })

        if token:
            self.session.headers['Authorization'] = f'token {token}'

        # Configure SSL
        self.session.verify = verify_ssl

        # Add retry adapter
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )

        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def safe_request(self, method, url, **kwargs):
        """Make a request with error handling"""
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                return response
            except requests.exceptions.SSLError as e:
                print(f"SSL Error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    print("SSL Error persists. Try running with --no-ssl-verify flag")
                    return None
            except requests.exceptions.ConnectionError as e:
                print(f"Connection Error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None

    def fetch_prs_simple(self, owner, repo, output_dir, max_prs=100):
        """Fetch PRs using simple REST API calls"""
        prs_dir = output_dir / 'prs'
        prs_dir.mkdir(exist_ok=True)
        
        # Check what we already have
        existing_prs = {int(f.stem.split('-')[1]) for f in prs_dir.glob('pr-*.json')}
        print(f"Found {len(existing_prs)} existing PRs")
        
        page = 1
        fetched = 0
        
        while fetched < max_prs:
            print(f"\nFetching PRs page {page}...")

            response = self.safe_request('GET',
                f'https://api.github.com/repos/{owner}/{repo}/pulls',
                params={
                    'state': 'all',
                    'per_page': 30,  # Smaller page size for reliability
                    'page': page,
                    'sort': 'updated',
                    'direction': 'desc'
                }
            )

            if not response or response.status_code != 200:
                print(f"Failed to fetch page {page}")
                break

            prs = response.json()
            if not prs:
                print("No more PRs to fetch")
                break

            for pr in prs:
                pr_number = pr['number']

                if pr_number in existing_prs:
                    print(f"  Skipping PR #{pr_number} (already exists)")
                    continue

                print(f"  Fetching PR #{pr_number}: {pr['title'][:50]}...")

                # Get PR details
                pr_data = self.fetch_pr_details_simple(owner, repo, pr_number)
                if pr_data:
                    # Save PR
                    pr_file = prs_dir / f'pr-{pr_number}.json'
                    with open(pr_file, 'w', encoding='utf-8') as f:
                        json.dump(pr_data, f, indent=2)

                    fetched += 1
                    print(f"    ✓ Saved PR #{pr_number}")

                    if fetched >= max_prs:
                        break

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            page += 1

        print(f"\nFetched {fetched} new PRs")
        return fetched

    def fetch_pr_details_simple(self, owner, repo, pr_number):
        """Fetch PR details using REST API"""
        # Get basic PR info
        pr_response = self.safe_request('GET',
            f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        )

        if not pr_response or pr_response.status_code != 200:
            return None

        pr_data = pr_response.json()

        # Format data to match expected structure
        formatted_data = {
            'number': pr_data['number'],
            'title': pr_data['title'],
            'body': pr_data['body'] or '',
            'state': pr_data['state'].upper(),
            'createdAt': pr_data['created_at'],
            'updatedAt': pr_data['updated_at'],
            'closedAt': pr_data['closed_at'],
            'mergedAt': pr_data['merged_at'],
            'author': {'login': pr_data['user']['login']} if pr_data.get('user') else None,
            'labels': [label['name'] for label in pr_data.get('labels', [])],
            'additions': pr_data.get('additions', 0),
            'deletions': pr_data.get('deletions', 0),
            'changedFiles': pr_data.get('changed_files', 0),
            'headRefName': pr_data.get('head', {}).get('ref', ''),
            'commits': {'nodes': []},
            'comments': {'nodes': []}
        }

        # Get commits (limited)
        commits_response = self.safe_request('GET',
            f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits',
            params={'per_page': 100}
        )

        if commits_response and commits_response.status_code == 200:
            commits = commits_response.json()
            formatted_data['commits']['nodes'] = [
                {
                    'commit': {
                        'oid': c['sha'],
                        'message': c['commit']['message'],
                        'committedDate': c['commit']['committer']['date'],
                        'author': {
                            'name': c['commit']['author']['name'],
                            'email': c['commit']['author']['email']
                        }
                    }
                }
                for c in commits
            ]

        # Get comments (limited)
        comments_response = self.safe_request('GET',
            f'https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments',
            params={'per_page': 100}
        )

        if comments_response and comments_response.status_code == 200:
            comments = comments_response.json()
            formatted_data['comments']['nodes'] = [
                {
                    'body': c['body'],
                    'createdAt': c['created_at'],
                    'author': {'login': c['user']['login']} if c.get('user') else None
                }
                for c in comments
            ]
        
        return formatted_data

    def fetch_issues_simple(self, owner, repo, output_dir, max_issues=100):
        """Fetch issues using simple REST API"""
        backlog_dir = output_dir / 'backlog'
        backlog_dir.mkdir(exist_ok=True)

        # Check existing
        existing_issues = {int(f.stem.split('-')[1]) for f in backlog_dir.glob('issue-*.json')}
        print(f"Found {len(existing_issues)} existing issues")

        page = 1
        fetched = 0

        while fetched < max_issues:
            print(f"\nFetching issues page {page}...")

            response = self.safe_request('GET',
                f'https://api.github.com/repos/{owner}/{repo}/issues',
                params={
                    'state': 'all',
                    'per_page': 30,
                    'page': page,
                    'sort': 'updated',
                    'direction': 'desc'
                }
            )

            if not response or response.status_code != 200:
                print(f"Failed to fetch page {page}")
                break

            issues = response.json()
            if not issues:
                print("No more issues")
                break

            for issue in issues:
                # Skip PRs
                if 'pull_request' in issue:
                    continue

                issue_number = issue['number']

                if issue_number in existing_issues:
                    print(f"  Skipping issue #{issue_number} (already exists)")
                    continue

                print(f"  Saving issue #{issue_number}: {issue['title'][:50]}...")

                # Format issue data
                issue_data = {
                    'number': issue['number'],
                    'title': issue['title'],
                    'body': issue['body'] or '',
                    'state': issue['state'].upper(),
                    'createdAt': issue['created_at'],
                    'updatedAt': issue['updated_at'],
                    'closedAt': issue['closed_at'],
                    'author': {'login': issue['user']['login']} if issue.get('user') else None,
                    'labels': [label['name'] for label in issue.get('labels', [])],
                    'comments': {'nodes': []}
                }

                # Save issue
                issue_file = backlog_dir / f'issue-{issue_number}.json'
                with open(issue_file, 'w', encoding='utf-8') as f:
                    json.dump(issue_data, f, indent=2)

                fetched += 1
                print(f"    ✓ Saved issue #{issue_number}")

                if fetched >= max_issues:
                    break

                time.sleep(0.3)

            page += 1

        print(f"\nFetched {fetched} new issues")
        return fetched

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust MSBuild data fetcher')
    parser.add_argument('--max-prs', type=int, default=100,
                       help='Maximum PRs to fetch (default: 100)')
    parser.add_argument('--max-issues', type=int, default=100,
                       help='Maximum issues to fetch (default: 100)')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Disable SSL verification (use for corporate networks)')
    args = parser.parse_args()
    
    # Check for token
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Warning: No GITHUB_TOKEN found.")
        print("You'll be limited to 60 requests per hour.")
        print("\nTo set token in PowerShell:")
        print('  $env:GITHUB_TOKEN="your_token_here"')
        print("\nTo create a token:")
        print("  1. Go to https://github.com/settings/tokens")
        print("  2. Generate new token (classic)")
        print("  3. Select 'repo' scope")
        print("")

        response = input("Continue without token? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / 'data' / 'raw-data' / 'msbuild'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fetcher
    verify_ssl = not args.no_ssl_verify
    if not verify_ssl:
        print("\n⚠️  SSL verification disabled!")
    
    fetcher = RobustFetcher(token, verify_ssl=verify_ssl)

    print(f"\nOutput directory: {output_dir}")
    print(f"Fetching up to {args.max_prs} PRs and {args.max_issues} issues...")

    # Fetch data
    pr_count = fetcher.fetch_prs_simple('dotnet', 'msbuild', output_dir, args.max_prs)
    issue_count = fetcher.fetch_issues_simple('dotnet', 'msbuild', output_dir, args.max_issues)

    # Summary
    print("\n" + "="*50)
    print("FETCH COMPLETE!")
    print("="*50)
    print(f"New PRs fetched: {pr_count}")
    print(f"New issues fetched: {issue_count}")

    # Count total files
    total_prs = len(list((output_dir / 'prs').glob('pr-*.json')))
    total_issues = len(list((output_dir / 'backlog').glob('issue-*.json')))

    print(f"\nTotal in dataset:")
    print(f"  PRs: {total_prs}")
    print(f"  Issues: {total_issues}")

    print("\nNext steps:")
    print("1. To fetch commit diffs, run: python fetch_diffs.py")
    print("2. To generate dataset, run: python scripts/model/generate_dataset.py")
    print("3. To analyze backlog, run: python scripts/model/analyze_msbuild_backlog.py")

if __name__ == '__main__':
    main()
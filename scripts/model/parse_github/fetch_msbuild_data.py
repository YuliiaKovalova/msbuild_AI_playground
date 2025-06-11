#!/usr/bin/env python3
import json
import requests
import time
from pathlib import Path
from datetime import datetime
import os
import sys

class GitHubDataFetcher:
    def __init__(self, token=None):
        self.token = token
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MSBuild-Dataset-Fetcher'
        }
        if token:
            self.headers['Authorization'] = f'token {token}'

        self.base_url = 'https://api.github.com'
        self.graphql_url = 'https://api.github.com/graphql'

    def check_rate_limit(self):
        """Check GitHub API rate limit"""
        response = requests.get(f'{self.base_url}/rate_limit', headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            core = data['rate']['remaining']
            reset_time = datetime.fromtimestamp(data['rate']['reset'])
            return core, reset_time
        return None, None
    
    def wait_if_rate_limited(self, response):
        """Wait if rate limited"""
        if response.status_code == 403:
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                wait_time = int(reset_time) - int(time.time()) + 10
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return True
        return False

    def fetch_pull_requests(self, owner, repo, output_dir, max_prs=100, state='all', since=None):
        """Fetch pull requests using REST API"""
        prs_dir = output_dir / 'prs'
        prs_dir.mkdir(exist_ok=True)
        
        # Check existing PRs to avoid refetching
        existing_prs = {int(f.stem.split('-')[1]) for f in prs_dir.glob('pr-*.json')}
        print(f"Found {len(existing_prs)} existing PRs in cache")
        
        page = 1
        fetched = 0
        skipped = 0
        
        while fetched < max_prs:
            print(f"Fetching PRs page {page}...")
            url = f'{self.base_url}/repos/{owner}/{repo}/pulls'
            params = {
                'state': state,
                'per_page': 100,  # Max allowed
                'page': page,
                'sort': 'updated',
                'direction': 'desc'
            }

            response = requests.get(url, headers=self.headers, params=params)

            if self.wait_if_rate_limited(response):
                continue

            if response.status_code != 200:
                print(f"Error fetching PRs: {response.status_code}")
                break

            prs = response.json()
            if not prs:
                print("No more PRs to fetch")
                break

            for pr in prs:
                pr_number = pr['number']

                # Skip if already fetched
                if pr_number in existing_prs:
                    skipped += 1
                    continue

                # Check date filter
                if since:
                    created_date = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                    since_date = datetime.fromisoformat(since)
                    if created_date < since_date:
                        print(f"Reached PRs older than {since}, stopping")
                        return fetched

                print(f"  Fetching detailed data for PR #{pr_number}...")

                # Fetch detailed PR data including commits and comments
                pr_detail = self.fetch_pr_details(owner, repo, pr_number)
                if pr_detail:
                    # Save PR data
                    pr_file = prs_dir / f'pr-{pr_number}.json'
                    with open(pr_file, 'w', encoding='utf-8') as f:
                        json.dump(pr_detail, f, indent=2)

                    # Fetch diffs for this PR
                    self.fetch_commit_diffs(owner, repo, pr_detail, output_dir)

                    fetched += 1
                    if fetched >= max_prs:
                        break

            page += 1
            time.sleep(0.5)  # Be nice to the API

        print(f"Fetched {fetched} new pull requests (skipped {skipped} existing)")
        return fetched

    def fetch_pr_details(self, owner, repo, pr_number):
        """Fetch detailed PR data using GraphQL for efficiency"""
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              number
              title
              body
              state
              createdAt
              updatedAt
              closedAt
              mergedAt
              additions
              deletions
              changedFiles
              headRefName
              isDraft
              author {
                login
              }
              milestone {
                title
              }
              assignees(first: 10) {
                nodes {
                  login
                }
              }
              labels(first: 20) {
                nodes {
                  name
                }
              }
              commits(first: 100) {
                nodes {
                  commit {
                    oid
                    message
                    committedDate
                    author {
                      name
                      email
                    }
                  }
                }
              }
              comments(first: 100) {
                nodes {
                  body
                  createdAt
                  author {
                    login
                  }
                }
              }
              reviewThreads(first: 50) {
                nodes {
                  comments(first: 50) {
                    nodes {
                      body
                      createdAt
                      path
                      diffHunk
                      author {
                        login
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        variables = {
            "owner": owner,
            "repo": repo,
            "number": pr_number
        }

        response = requests.post(
            self.graphql_url,
            headers=self.headers,
            json={'query': query, 'variables': variables}
        )

        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']['repository']['pullRequest']:
                pr_data = data['data']['repository']['pullRequest']
                # Transform labels to simple list
                if 'labels' in pr_data and pr_data['labels']['nodes']:
                    pr_data['labels'] = [label['name'] for label in pr_data['labels']['nodes']]
                else:
                    pr_data['labels'] = []
                return pr_data
        else:
            print(f"Error fetching PR #{pr_number}: {response.status_code}")

        return None

    def fetch_commit_diffs(self, owner, repo, pr_data, output_dir):
        """Fetch diffs for commits in a PR"""
        diffs_dir = output_dir / 'diffs'
        diffs_dir.mkdir(exist_ok=True)

        if 'commits' not in pr_data or 'nodes' not in pr_data['commits']:
            return

        for node in pr_data['commits']['nodes']:
            commit = node.get('commit', {})
            oid = commit.get('oid')
            if not oid:
                continue

            diff_file = diffs_dir / f'{oid}.diff'
            if diff_file.exists():
                continue

            print(f"    Fetching diff for commit {oid[:8]}...")
            url = f'{self.base_url}/repos/{owner}/{repo}/commits/{oid}'
            headers = self.headers.copy()
            headers['Accept'] = 'application/vnd.github.v3.diff'

            response = requests.get(url, headers=headers)

            if self.wait_if_rate_limited(response):
                response = requests.get(url, headers=headers)

            if response.status_code == 200:
                with open(diff_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                print(f"      Error fetching diff: {response.status_code}")

            time.sleep(0.2)

    def fetch_issues(self, owner, repo, output_dir, max_issues=100, state='all', since=None):
        """Fetch issues (backlog items)"""
        backlog_dir = output_dir / 'backlog'
        backlog_dir.mkdir(exist_ok=True)

        # Check existing issues to avoid refetching
        existing_issues = {int(f.stem.split('-')[1]) for f in backlog_dir.glob('issue-*.json')}
        print(f"Found {len(existing_issues)} existing issues in cache")

        page = 1
        fetched = 0
        skipped = 0

        while fetched < max_issues:
            print(f"Fetching issues page {page}...")
            url = f'{self.base_url}/repos/{owner}/{repo}/issues'
            params = {
                'state': state,
                'per_page': 100,  # Max allowed
                'page': page,
                'sort': 'updated',
                'direction': 'desc'
            }

            response = requests.get(url, headers=self.headers, params=params)

            if self.wait_if_rate_limited(response):
                continue

            if response.status_code != 200:
                print(f"Error fetching issues: {response.status_code}")
                break

            issues = response.json()
            if not issues:
                print("No more issues to fetch")
                break

            for issue in issues:
                # Skip pull requests
                if 'pull_request' in issue:
                    continue

                issue_number = issue['number']

                # Skip if already fetched
                if issue_number in existing_issues:
                    skipped += 1
                    continue

                # Check date filter
                if since:
                    created_date = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                    since_date = datetime.fromisoformat(since)
                    if created_date < since_date:
                        print(f"Reached issues older than {since}, stopping")
                        return fetched

                print(f"  Fetching detailed data for issue #{issue_number}...")

                # Fetch detailed issue data
                issue_detail = self.fetch_issue_details(owner, repo, issue_number)
                if issue_detail:
                    # Save issue data
                    issue_file = backlog_dir / f'issue-{issue_number}.json'
                    with open(issue_file, 'w', encoding='utf-8') as f:
                        json.dump(issue_detail, f, indent=2)

                    fetched += 1
                    if fetched >= max_issues:
                        break

            page += 1
            time.sleep(0.5)

        print(f"Fetched {fetched} new issues (skipped {skipped} existing)")
        return fetched

    def fetch_issue_details(self, owner, repo, issue_number):
        """Fetch detailed issue data using GraphQL"""
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) {
              number
              title
              body
              state
              createdAt
              updatedAt
              closedAt
              author {
                login
              }
              milestone {
                title
              }
              assignees(first: 10) {
                nodes {
                  login
                }
              }
              labels(first: 20) {
                nodes {
                  name
                }
              }
              comments(first: 100) {
                nodes {
                  body
                  createdAt
                  author {
                    login
                  }
                }
              }
            }
          }
        }
        """

        variables = {
            "owner": owner,
            "repo": repo,
            "number": issue_number
        }

        response = requests.post(
            self.graphql_url,
            headers=self.headers,
            json={'query': query, 'variables': variables}
        )

        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']['repository']['issue']:
                issue_data = data['data']['repository']['issue']
                # Transform labels to simple list
                if 'labels' in issue_data and issue_data['labels']['nodes']:
                    issue_data['labels'] = [label['name'] for label in issue_data['labels']['nodes']]
                else:
                    issue_data['labels'] = []
                return issue_data
        else:
            print(f"Error fetching issue #{issue_number}: {response.status_code}")

        return None

def main():
    # Configuration
    OWNER = 'dotnet'
    REPO = 'msbuild'
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fetch MSBuild repository data')
    parser.add_argument('--max-prs', type=int, default=500, help='Maximum number of PRs to fetch (default: 500)')
    parser.add_argument('--max-issues', type=int, default=500, help='Maximum number of issues to fetch (default: 500)')
    parser.add_argument('--since', type=str, help='Fetch items since date (YYYY-MM-DD)')
    parser.add_argument('--state', type=str, default='all', choices=['all', 'open', 'closed'], help='State of items to fetch')
    args = parser.parse_args()
    
    MAX_PRS = args.max_prs
    MAX_ISSUES = args.max_issues
    
    # Get GitHub token from environment variable
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Warning: No GITHUB_TOKEN found. API rate limits will be very restrictive.")
        print("Set your token with: $env:GITHUB_TOKEN='your_token_here'")
        response = input("Continue without token? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup output directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / 'data' / 'raw-data' / 'msbuild'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize fetcher
    fetcher = GitHubDataFetcher(token)
    
    # Check rate limit
    remaining, reset_time = fetcher.check_rate_limit()
    if remaining is not None:
        print(f"API rate limit: {remaining} requests remaining")
        if reset_time:
            print(f"Rate limit resets at: {reset_time}")

    # Fetch pull requests
    print(f"\nFetching up to {MAX_PRS} pull requests (state: {args.state})...")
    if args.since:
        print(f"Filtering to items created since: {args.since}")
    pr_count = fetcher.fetch_pull_requests(OWNER, REPO, output_dir, MAX_PRS, state=args.state, since=args.since)

    # Fetch issues
    print(f"\nFetching up to {MAX_ISSUES} issues (state: {args.state})...")
    issue_count = fetcher.fetch_issues(OWNER, REPO, output_dir, MAX_ISSUES, state=args.state, since=args.since)

    print(f"\nData fetching complete!")
    print(f"Total PRs in dataset: {len(list((output_dir / 'prs').glob('pr-*.json')))}")
    print(f"Total issues in dataset: {len(list((output_dir / 'backlog').glob('issue-*.json')))}")
    print(f"Total diffs in dataset: {len(list((output_dir / 'diffs').glob('*.diff')))}")
    print(f"Data saved to: {output_dir}")

    # Show sample commands for next steps
    print("\nNext steps:")
    print("1. To fetch more data: python scripts/model/fetch_msbuild_data.py --max-prs 1000 --max-issues 1000")
    print("2. To generate dataset: python scripts/model/generate_dataset.py")

if __name__ == '__main__':
    main()
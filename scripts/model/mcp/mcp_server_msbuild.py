#!/usr/bin/env python3
"""
MCP Server for MSBuild Repository Analytics
Provides real-time statistics and insights about the MSBuild repository
"""

import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from collections import defaultdict
import re

class MSBuildAnalyticsServer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "msbuild_analytics.db"
        self.server = Server("msbuild-analytics")
        self.conn = None
        
        # Register handlers
        self._register_handlers()
        
    def _register_handlers(self):
        """Register all MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="get_statistics",
                    description="Get overall repository statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_trends": {
                                "type": "boolean",
                                "description": "Include trend analysis",
                                "default": True
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_issue_stats",
                    description="Get detailed issue statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "state": {
                                "type": "string",
                                "enum": ["OPEN", "CLOSED", "ALL"],
                                "description": "Filter by issue state",
                                "default": "ALL"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Analyze issues from last N days",
                                "default": 30
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_pr_stats",
                    description="Get pull request statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "state": {
                                "type": "string",
                                "enum": ["OPEN", "MERGED", "CLOSED", "ALL"],
                                "default": "ALL"
                            },
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include detailed metrics",
                                "default": True
                            }
                        }
                    }
                ),
                types.Tool(
                    name="find_duplicates",
                    description="Find duplicate issues",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "threshold": {
                                "type": "number",
                                "description": "Similarity threshold (0-1)",
                                "default": 0.8
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of groups to return",
                                "default": 10
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_trending_topics",
                    description="Get trending topics and categories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "Analyze trends from last N days",
                                "default": 7
                            }
                        }
                    }
                ),
                types.Tool(
                    name="search_issues",
                    description="Search issues by keywords or criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by labels"
                            },
                            "state": {
                                "type": "string",
                                "enum": ["OPEN", "CLOSED", "ALL"],
                                "default": "ALL"
                            },
                            "limit": {
                                "type": "integer",
                                "default": 20
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="get_contributor_stats",
                    description="Get contributor statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of top contributors to show",
                                "default": 10
                            }
                        }
                    }
                )
            ]
        
        # ...existing code...

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            
            print(f"[DEBUG] Tool called: {name} with args: {arguments}")  # Add debug line
            
            try:
                if not self.conn:
                    await self._initialize_db()

                args = arguments or {}

                if name == "get_statistics":
                    result = await self._get_statistics(args.get("include_trends", True))
                elif name == "get_issue_stats":
                    result = await self._get_issue_stats(
                        args.get("state", "ALL"),
                        args.get("days", 30)
                    )
                elif name == "get_pr_stats":
                    result = await self._get_pr_stats(
                        args.get("state", "ALL"),
                        args.get("include_metrics", True)
                    )
                elif name == "find_duplicates":
                    result = await self._find_duplicates(
                        args.get("threshold", 0.8),
                        args.get("limit", 1000)
                    )
                elif name == "get_trending_topics":
                    result = await self._get_trending_topics(args.get("days", 1000))
                elif name == "search_issues":
                    result = await self._search_issues(
                        args.get("query", ""),
                        args.get("labels", []),
                        args.get("state", "ALL"),
                        args.get("limit", 200)
                    )
                elif name == "get_contributor_stats":
                    result = await self._get_contributor_stats(args.get("top_n", 10))
                else:
                    result = {"error": f"Unknown tool: {name}"}

                print(f"[DEBUG] Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")  # Add debug line

                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                print(f"[DEBUG] Error in handle_call_tool: {type(e).__name__}: {str(e)}")  # Add debug line
                import traceback
                traceback.print_exc()

                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": str(e), "type": type(e).__name__}, indent=2)
                )]
    
    async def _initialize_db(self):
        """Initialize SQLite database with converted data"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Check if database is already initialized
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='issues'")
        if cursor.fetchone():
            return
        
        print("Initializing database...")
        await self._create_schema()
        await self._import_data()
        print("Database initialized!")
    
    async def _create_schema(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY,
                number INTEGER UNIQUE,
                title TEXT,
                body TEXT,
                state TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                closed_at TIMESTAMP,
                author TEXT,
                labels TEXT,
                comments_count INTEGER,
                milestone TEXT
            )
        """)
        
        # PRs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pull_requests (
                id INTEGER PRIMARY KEY,
                number INTEGER UNIQUE,
                title TEXT,
                body TEXT,
                state TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                merged_at TIMESTAMP,
                closed_at TIMESTAMP,
                author TEXT,
                labels TEXT,
                commits_count INTEGER,
                additions INTEGER,
                deletions INTEGER,
                changed_files INTEGER
            )
        """)
        
        # Comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY,
                issue_number INTEGER,
                pr_number INTEGER,
                author TEXT,
                body TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (issue_number) REFERENCES issues(number),
                FOREIGN KEY (pr_number) REFERENCES pull_requests(number)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_state ON issues(state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_created ON issues(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prs_state ON pull_requests(state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prs_created ON pull_requests(created_at)")
        
        self.conn.commit()
    
    async def _import_data(self):
        """Import data from JSON files to SQLite"""
        cursor = self.conn.cursor()
        
        # Try different path structures
        possible_paths = [
            self.data_dir / "raw-data" / "msbuild",
            self.data_dir / "msbuild",
            self.data_dir
        ]
        
        issues_dir = None
        prs_dir = None
        
        for base_path in possible_paths:
            if (base_path / "backlog").exists():
                issues_dir = base_path / "backlog"
                prs_dir = base_path / "prs"
                print(f"Found data in: {base_path}")
                break
        
        if not issues_dir:
            print(f"ERROR: Could not find data directories!")
            print(f"Searched in: {[str(p) for p in possible_paths]}")
            return
        
        # Import issues
        if issues_dir.exists():
            print("Importing issues...")
            issue_count = 0
            
            for issue_file in issues_dir.glob("issue-*.json"):
                try:
                    with open(issue_file, 'r', encoding='utf-8') as f:
                        issue = json.load(f)
                    
                    labels = json.dumps(issue.get('labels', []))
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO issues 
                        (number, title, body, state, created_at, updated_at, closed_at,
                         author, labels, comments_count, milestone)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        issue.get('number'),
                        issue.get('title'),
                        issue.get('body'),
                        issue.get('state'),
                        issue.get('createdAt'),
                        issue.get('updatedAt'),
                        issue.get('closedAt'),
                        issue.get('author', {}).get('login') if issue.get('author') else None,
                        labels,
                        len(issue.get('comments', {}).get('nodes', [])),
                        issue.get('milestone', {}).get('title') if issue.get('milestone') else None
                    ))
                    
                    # Import comments
                    if 'comments' in issue and 'nodes' in issue['comments']:
                        for comment in issue['comments']['nodes']:
                            if comment:
                                cursor.execute("""
                                    INSERT INTO comments (issue_number, author, body, created_at)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    issue['number'],
                                    comment.get('author', {}).get('login') if comment.get('author') else None,
                                    comment.get('body'),
                                    comment.get('createdAt')
                                ))
                    
                    issue_count += 1
                    if issue_count % 100 == 0:
                        print(f"  Imported {issue_count} issues...")
                        self.conn.commit()
                
                except Exception as e:
                    print(f"Error importing {issue_file}: {e}")
            
            print(f"  Total issues imported: {issue_count}")
        
        # Import PRs
        prs_dir = self.data_dir / "raw-data" / "msbuild" / "prs"
        if prs_dir.exists():
            print("Importing pull requests...")
            pr_count = 0
            
            for pr_file in prs_dir.glob("pr-*.json"):
                try:
                    with open(pr_file, 'r', encoding='utf-8') as f:
                        pr = json.load(f)
                    
                    labels = json.dumps(pr.get('labels', []))
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO pull_requests 
                        (number, title, body, state, created_at, updated_at, merged_at, closed_at,
                         author, labels, commits_count, additions, deletions, changed_files)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pr.get('number'),
                        pr.get('title'),
                        pr.get('body'),
                        pr.get('state'),
                        pr.get('createdAt'),
                        pr.get('updatedAt'),
                        pr.get('mergedAt'),
                        pr.get('closedAt'),
                        pr.get('author', {}).get('login') if pr.get('author') else None,
                        labels,
                        len(pr.get('commits', {}).get('nodes', [])),
                        pr.get('additions', 0),
                        pr.get('deletions', 0),
                        pr.get('changedFiles', 0)
                    ))
                    
                    pr_count += 1
                    if pr_count % 50 == 0:
                        print(f"  Imported {pr_count} PRs...")
                        self.conn.commit()
                
                except Exception as e:
                    print(f"Error importing {pr_file}: {e}")
            
            print(f"  Total PRs imported: {pr_count}")
        
        self.conn.commit()
    
    async def _get_statistics(self, include_trends: bool) -> Dict[str, Any]:
        """Get overall repository statistics"""
        cursor = self.conn.cursor()
        
        # Basic counts
        stats = {
            "overview": {},
            "issues": {},
            "pull_requests": {}
        }
        
        # Issue statistics
        cursor.execute("SELECT COUNT(*) as total FROM issues")
        stats["issues"]["total"] = cursor.fetchone()["total"]
        
        cursor.execute("SELECT state, COUNT(*) as count FROM issues GROUP BY state")
        issue_states = {row["state"]: row["count"] for row in cursor.fetchall()}
        stats["issues"]["by_state"] = issue_states
        
        # PR statistics
        cursor.execute("SELECT COUNT(*) as total FROM pull_requests")
        stats["pull_requests"]["total"] = cursor.fetchone()["total"]
        
        cursor.execute("SELECT state, COUNT(*) as count FROM pull_requests GROUP BY state")
        pr_states = {row["state"]: row["count"] for row in cursor.fetchall()}
        stats["pull_requests"]["by_state"] = pr_states
        
        # Overview
        stats["overview"]["total_items"] = stats["issues"]["total"] + stats["pull_requests"]["total"]
        stats["overview"]["open_items"] = issue_states.get("OPEN", 0) + pr_states.get("OPEN", 0)
        
        if include_trends:
            # Recent activity (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM issues 
                WHERE created_at > ? AND state = 'OPEN'
            """, (thirty_days_ago,))
            stats["issues"]["opened_last_30_days"] = cursor.fetchone()["count"]
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM issues 
                WHERE closed_at > ? AND state = 'CLOSED'
            """, (thirty_days_ago,))
            stats["issues"]["closed_last_30_days"] = cursor.fetchone()["count"]
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM pull_requests 
                WHERE merged_at > ?
            """, (thirty_days_ago,))
            stats["pull_requests"]["merged_last_30_days"] = cursor.fetchone()["count"]
        
        # Top contributors
        cursor.execute("""
            SELECT author, COUNT(*) as count 
            FROM (
                SELECT author FROM issues WHERE author IS NOT NULL
                UNION ALL
                SELECT author FROM pull_requests WHERE author IS NOT NULL
            )
            GROUP BY author
            ORDER BY count DESC
            LIMIT 5
        """)
        stats["top_contributors"] = [
            {"author": row["author"], "contributions": row["count"]} 
            for row in cursor.fetchall()
        ]
        
        return stats
    
    async def _get_issue_stats(self, state: str, days: int) -> Dict[str, Any]:
        """Get detailed issue statistics"""
        cursor = self.conn.cursor()
        
        stats = {
            "state_filter": state,
            "time_period_days": days,
            "counts": {},
            "labels": {},
            "activity": {}
        }
        
        # Build query conditions
        conditions = []
        params = []
        
        if state != "ALL":
            conditions.append("state = ?")
            params.append(state)
        
        if days > 0:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            conditions.append("created_at > ?")
            params.append(cutoff_date)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Get counts
        cursor.execute(f"SELECT COUNT(*) as count FROM issues{where_clause}", params)
        stats["counts"]["total"] = cursor.fetchone()["count"]
        
        # Label distribution
        cursor.execute(f"""
            SELECT labels FROM issues{where_clause}
        """, params)
        
        label_counts = defaultdict(int)
        for row in cursor.fetchall():
            if row["labels"]:
                labels = json.loads(row["labels"])
                for label in labels:
                    if isinstance(label, str):
                        label_counts[label] += 1
                    elif isinstance(label, dict) and 'name' in label:
                        label_counts[label['name']] += 1
        
        stats["labels"] = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Activity metrics
        cursor.execute(f"""
            SELECT 
                AVG(comments_count) as avg_comments,
                MAX(comments_count) as max_comments
            FROM issues{where_clause}
        """, params)
        
        row = cursor.fetchone()
        stats["activity"]["average_comments"] = round(row["avg_comments"] or 0, 2)
        stats["activity"]["max_comments"] = row["max_comments"] or 0
        
        # Time to close for closed issues
        if state in ["CLOSED", "ALL"]:
            close_conditions = conditions.copy()
            close_params = params.copy()
            
            if state == "ALL":
                close_conditions.append("state = 'CLOSED'")
            
            close_where = " WHERE " + " AND ".join(close_conditions) if close_conditions else ""
            
            cursor.execute(f"""
                SELECT 
                    AVG(julianday(closed_at) - julianday(created_at)) as avg_days_to_close
                FROM issues{close_where}
                WHERE closed_at IS NOT NULL
            """, close_params)
            
            avg_days = cursor.fetchone()["avg_days_to_close"]
            if avg_days:
                stats["activity"]["average_days_to_close"] = round(avg_days, 1)
        
        return stats
    
    async def _get_pr_stats(self, state: str, include_metrics: bool) -> Dict[str, Any]:
        """Get pull request statistics"""
        cursor = self.conn.cursor()
        
        stats = {
            "state_filter": state,
            "counts": {},
            "metrics": {}
        }
        
        # Build query conditions
        conditions = []
        params = []
        
        if state != "ALL":
            conditions.append("state = ?")
            params.append(state)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Get counts
        cursor.execute(f"SELECT COUNT(*) as count FROM pull_requests{where_clause}", params)
        stats["counts"]["total"] = cursor.fetchone()["count"]
        
        if include_metrics:
            # Code change metrics
            cursor.execute(f"""
                SELECT 
                    AVG(additions) as avg_additions,
                    AVG(deletions) as avg_deletions,
                    AVG(changed_files) as avg_changed_files,
                    MAX(additions) as max_additions,
                    MAX(deletions) as max_deletions,
                    MAX(changed_files) as max_changed_files
                FROM pull_requests{where_clause}
            """, params)
            
            row = cursor.fetchone()
            stats["metrics"]["average"] = {
                "additions": round(row["avg_additions"] or 0, 1),
                "deletions": round(row["avg_deletions"] or 0, 1),
                "changed_files": round(row["avg_changed_files"] or 0, 1)
            }
            stats["metrics"]["maximum"] = {
                "additions": row["max_additions"] or 0,
                "deletions": row["max_deletions"] or 0,
                "changed_files": row["max_changed_files"] or 0
            }
            
            # Time to merge for merged PRs
            if state in ["MERGED", "ALL"]:
                merge_conditions = conditions.copy()
                merge_params = params.copy()
                
                if state == "ALL":
                    merge_conditions.append("state = 'MERGED'")
                
                merge_where = " WHERE " + " AND ".join(merge_conditions) if merge_conditions else ""
                
                cursor.execute(f"""
                    SELECT 
                        AVG(julianday(merged_at) - julianday(created_at)) as avg_days_to_merge,
                        MIN(julianday(merged_at) - julianday(created_at)) as min_days_to_merge,
                        MAX(julianday(merged_at) - julianday(created_at)) as max_days_to_merge
                    FROM pull_requests{merge_where}
                    WHERE merged_at IS NOT NULL
                """, merge_params)
                
                row = cursor.fetchone()
                if row["avg_days_to_merge"]:
                    stats["metrics"]["merge_time"] = {
                        "average_days": round(row["avg_days_to_merge"], 1),
                        "minimum_days": round(row["min_days_to_merge"], 1),
                        "maximum_days": round(row["max_days_to_merge"], 1)
                    }
        
        return stats
    
    async def _find_duplicates(self, threshold: float, limit: int) -> Dict[str, Any]:
        """Find duplicate issues using similarity analysis"""
        cursor = self.conn.cursor()
        
        # Get open issues
        cursor.execute("""
            SELECT number, title, body 
            FROM issues 
            WHERE state = 'OPEN'
        """)
        
        issues = []
        for row in cursor.fetchall():
            issues.append({
                "number": row["number"],
                "title": row["title"],
                "text": f"{row['title']} {row['body'] or ''}"
            })
        
        # Simple duplicate detection based on title similarity
        duplicates = []
        processed = set()
        
        for i, issue1 in enumerate(issues):
            if issue1["number"] in processed:
                continue
                
            similar = []
            for j, issue2 in enumerate(issues[i+1:], i+1):
                if issue2["number"] in processed:
                    continue
                
                # Simple similarity check (can be enhanced with TF-IDF)
                similarity = self._calculate_simple_similarity(issue1["text"], issue2["text"])
                
                if similarity > threshold:
                    similar.append({
                        "number": issue2["number"],
                        "title": issue2["title"],
                        "similarity": round(similarity, 2)
                    })
                    processed.add(issue2["number"])
            
            if similar:
                duplicates.append({
                    "primary": {
                        "number": issue1["number"],
                        "title": issue1["title"]
                    },
                    "duplicates": similar,
                    "total": len(similar) + 1
                })
                processed.add(issue1["number"])
        
        # Sort by group size and limit
        duplicates.sort(key=lambda x: x["total"], reverse=True)
        
        return {
            "duplicate_groups": duplicates[:limit],
            "total_groups": len(duplicates),
            "total_duplicate_issues": len(processed)
        }
    
    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _get_trending_topics(self, days: int) -> Dict[str, Any]:
        """Get trending topics based on recent activity"""
        cursor = self.conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get recent issues and their content
        cursor.execute("""
            SELECT number, title, body, labels, created_at
            FROM issues
            WHERE created_at > ?
            ORDER BY created_at DESC
        """, (cutoff_date,))
        
        # Extract topics from titles and labels
        topic_counts = defaultdict(int)
        recent_issues = []
        
        # Common topic patterns
        topic_patterns = {
            "build": re.compile(r'\b(build|compile|msbuild|compilation)\b', re.I),
            "performance": re.compile(r'\b(performance|slow|speed|optimize)\b', re.I),
            "error": re.compile(r'\b(error|exception|crash|fail)\b', re.I),
            "feature": re.compile(r'\b(feature|enhancement|add|new)\b', re.I),
            "bug": re.compile(r'\b(bug|fix|issue|problem)\b', re.I),
            "test": re.compile(r'\b(test|testing|unit|coverage)\b', re.I),
            "dependency": re.compile(r'\b(dependency|package|nuget|reference)\b', re.I),
            "documentation": re.compile(r'\b(doc|documentation|readme|help)\b', re.I)
        }
        
        for row in cursor.fetchall():
            text = f"{row['title']} {row['body'] or ''}"
            issue_topics = []
            
            # Check patterns
            for topic, pattern in topic_patterns.items():
                if pattern.search(text):
                    topic_counts[topic] += 1
                    issue_topics.append(topic)
            
            # Check labels
            if row["labels"]:
                labels = json.loads(row["labels"])
                for label in labels:
                    label_name = label if isinstance(label, str) else label.get('name', '')
                    if label_name:
                        topic_counts[f"label:{label_name}"] += 1
            
            if issue_topics:
                recent_issues.append({
                    "number": row["number"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "topics": issue_topics
                })
        
        # Sort topics by frequency
        trending = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_period_days": days,
            "trending_topics": [
                {"topic": topic, "count": count} for topic, count in trending
            ],
            "recent_examples": recent_issues[:5]
        }
    
    async def _search_issues(self, query: str, labels: List[str], state: str, limit: int) -> Dict[str, Any]:
        """Search issues by query and filters"""
        cursor = self.conn.cursor()
        
        conditions = []
        params = []
        
        # State filter
        if state != "ALL":
            conditions.append("state = ?")
            params.append(state)
        
        # Text search in title and body
        if query:
            conditions.append("(title LIKE ? OR body LIKE ?)")
            query_param = f"%{query}%"
            params.extend([query_param, query_param])
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        cursor.execute(f"""
            SELECT number, title, state, labels, created_at, updated_at, author
            FROM issues{where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """, params + [limit])
        
        results = []
        for row in cursor.fetchall():
            # Check label filter if specified
            if labels:
                issue_labels = json.loads(row["labels"]) if row["labels"] else []
                issue_label_names = [
                    l if isinstance(l, str) else l.get('name', '') 
                    for l in issue_labels
                ]
                
                if not any(label in issue_label_names for label in labels):
                    continue
            
            results.append({
                "number": row["number"],
                "title": row["title"],
                "state": row["state"],
                "author": row["author"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "labels": json.loads(row["labels"]) if row["labels"] else []
            })
        
        return {
            "query": query,
            "filters": {
                "state": state,
                "labels": labels
            },
            "count": len(results),
            "results": results
        }
    
    async def _get_contributor_stats(self, top_n: int) -> Dict[str, Any]:
        """Get contributor statistics"""
        cursor = self.conn.cursor()
        
        # Get top contributors by issues created
        cursor.execute("""
            SELECT author, COUNT(*) as issue_count
            FROM issues
            WHERE author IS NOT NULL
            GROUP BY author
            ORDER BY issue_count DESC
            LIMIT ?
        """, (top_n,))
        
        issue_contributors = {row["author"]: row["issue_count"] for row in cursor.fetchall()}
        
        # Get top contributors by PRs created
        cursor.execute("""
            SELECT author, COUNT(*) as pr_count
            FROM pull_requests
            WHERE author IS NOT NULL
            GROUP BY author
            ORDER BY pr_count DESC
            LIMIT ?
        """, (top_n,))
        
        pr_contributors = {row["author"]: row["pr_count"] for row in cursor.fetchall()}
        
        # Get merged PR contributors
        cursor.execute("""
            SELECT author, COUNT(*) as merged_count
            FROM pull_requests
            WHERE author IS NOT NULL AND state = 'MERGED'
            GROUP BY author
            ORDER BY merged_count DESC
            LIMIT ?
        """, (top_n,))
        
        merged_contributors = {row["author"]: row["merged_count"] for row in cursor.fetchall()}
        
        # Combine all contributors
        all_contributors = set(issue_contributors.keys()) | set(pr_contributors.keys())
        
        contributor_stats = []
        for author in all_contributors:
            stats = {
                "author": author,
                "issues_created": issue_contributors.get(author, 0),
                "prs_created": pr_contributors.get(author, 0),
                "prs_merged": merged_contributors.get(author, 0),
                "total_contributions": (
                    issue_contributors.get(author, 0) + 
                    pr_contributors.get(author, 0)
                )
            }
            contributor_stats.append(stats)
        
        # Sort by total contributions
        contributor_stats.sort(key=lambda x: x["total_contributions"], reverse=True)
        
        return {
            "top_contributors": contributor_stats[:top_n],
            "total_contributors": len(all_contributors)
        }
    
    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="msbuild-analytics",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mcp_server_msbuild.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    server = MSBuildAnalyticsServer(data_dir)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
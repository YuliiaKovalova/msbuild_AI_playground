#!/usr/bin/env python3
"""
Example MCP Client for MSBuild Analytics
Shows how to interact with the MCP server
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MSBuildAnalyticsClient:
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        
# ...existing code...

   # ...existing code...

    async def run_query(self, tool_name: str, arguments: dict = None):
        """Run a query against the MCP server"""
        import os

        # Get the full path to the server script
        server_path = os.path.join(os.path.dirname(__file__), self.server_script_path)
        data_path = r"c:\Users\ykovalova\msbuild\data"

        server_params = StdioServerParameters(
            command="python",
            args=[server_path, data_path],
            env=None
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()

                    # List available tools (for debugging)
                    try:
                        tools_response = await session.list_tools()
                        # Handle different response formats
                        if hasattr(tools_response, 'tools'):
                            tools = tools_response.tools
                        elif isinstance(tools_response, (list, tuple)) and len(tools_response) > 0:
                            tools = tools_response[0] if isinstance(tools_response[0], list) else tools_response
                        else:
                            tools = tools_response

                        if tools:
                            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tools]
                            print(f"Available tools: {tool_names}")
                    except:
                        print("Could not list tools")

                    # Call the tool
                    result = await session.call_tool(tool_name, arguments or {})

                    # Extract the text content from the result
                    if result:
                        if hasattr(result, 'content'):
                            for content in result.content:
                                if hasattr(content, 'text'):
                                    return json.loads(content.text)
                        elif isinstance(result, list):
                            for item in result:
                                if hasattr(item, 'text'):
                                    return json.loads(item.text)

                    return None

        except Exception as e:
            print(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

async def demo_queries():
    """Demonstrate various queries"""
    client = MSBuildAnalyticsClient("mcp_server_msbuild.py")
    
    print("MSBuild Analytics MCP Client Demo")
    print("=" * 50)
    
    # Query 1: Get overall statistics
    print("\n1. Getting overall repository statistics...")
    stats = await client.run_query("get_statistics", {"include_trends": True})
    if stats:
        print(f"\nRepository Overview:")
        print(f"  Total items: {stats['overview']['total_items']}")
        print(f"  Open items: {stats['overview']['open_items']}")
        print(f"\nIssue Statistics:")
        print(f"  Total issues: {stats['issues']['total']}")
        print(f"  Open issues: {stats['issues']['by_state'].get('OPEN', 0)}")
        print(f"  Closed issues: {stats['issues']['by_state'].get('CLOSED', 0)}")
    
    # Query 2: Get issue statistics for last 30 days
    print("\n2. Getting issue statistics for last 30 days...")
    issue_stats = await client.run_query("get_issue_stats", {
        "state": "OPEN",
        "days": 30
    })
    if issue_stats:
        print(f"\nOpen Issues (Last 30 days):")
        print(f"  Total: {issue_stats['counts']['total']}")
        print(f"  Average comments: {issue_stats['activity']['average_comments']}")
        print(f"\nTop Labels:")
        for label, count in list(issue_stats['labels'].items())[:5]:
            print(f"    {label}: {count}")
    
    # Query 3: Find duplicate issues
    print("\n3. Finding duplicate issues...")
    duplicates = await client.run_query("find_duplicates", {
        "threshold": 0.7,
        "limit": 5
    })
    if duplicates:
        print(f"\nFound {duplicates['total_groups']} duplicate groups")
        print(f"Total duplicate issues: {duplicates['total_duplicate_issues']}")
        for i, group in enumerate(duplicates['duplicate_groups'][:3]):
            print(f"\n  Group {i+1} ({group['total']} issues):")
            print(f"    Primary: #{group['primary']['number']} - {group['primary']['title']}")
            for dup in group['duplicates']:
                print(f"    Duplicate: #{dup['number']} (similarity: {dup['similarity']})")
    
    # Query 4: Get trending topics
    print("\n4. Getting trending topics...")
    trends = await client.run_query("get_trending_topics", {"days": 7})
    if trends:
        print(f"\nTrending Topics (Last 7 days):")
        for topic in trends['trending_topics'][:5]:
            print(f"  {topic['topic']}: {topic['count']} mentions")
    
    # Query 5: Search for specific issues
    print("\n5. Searching for performance-related issues...")
    search_results = await client.run_query("search_issues", {
        "query": "performance",
        "state": "OPEN",
        "limit": 5
    })
    if search_results:
        print(f"\nFound {search_results['count']} matching issues:")
        for issue in search_results['results'][:3]:
            print(f"  #{issue['number']}: {issue['title']}")
            print(f"    State: {issue['state']}, Author: {issue['author']}")
    
    # Query 6: Get contributor statistics
    print("\n6. Getting top contributors...")
    contributors = await client.run_query("get_contributor_stats", {"top_n": 5})
    if contributors:
        print(f"\nTop Contributors (Total: {contributors['total_contributors']}):")
        for contrib in contributors['top_contributors']:
            print(f"  {contrib['author']}:")
            print(f"    Issues: {contrib['issues_created']}, PRs: {contrib['prs_created']}, Merged: {contrib['prs_merged']}")

async def interactive_mode():
    """Interactive query mode"""
    client = MSBuildAnalyticsClient("mcp_server_msbuild.py")
    
    print("\nMSBuild Analytics Interactive Mode")
    print("Type 'help' for available commands, 'quit' to exit")
    
    commands = {
        "stats": ("get_statistics", {"include_trends": True}),
        "issues": ("get_issue_stats", {"state": "OPEN", "days": 1000}),
        "prs": ("get_pr_stats", {"state": "ALL", "include_metrics": True}),
        "duplicates": ("find_duplicates", {"threshold": 0.8, "limit": 100}),
        "trends": ("get_trending_topics", {"days": 7}),
        "contributors": ("get_contributor_stats", {"top_n": 10}),
        "stale": ("search_issues", {"query": "", "state": "OPEN", "limit": 200})  # Add this
    }
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("\nAvailable commands:")
                for cmd in commands:
                    print(f"  {cmd}")
                print("  search <query> - Search for issues")
                print("  quit - Exit")
            elif command.startswith("search "):
                query = command[7:]
                result = await client.run_query("search_issues", {
                    "query": query,
                    "state": "ALL",
                    "limit": 10
                })
                if result:
                    print(json.dumps(result, indent=2))
            elif command in commands:
                tool_name, args = commands[command]
                result = await client.run_query(tool_name, args)
                if result:
                    print(json.dumps(result, indent=2))
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await interactive_mode()
    else:
        await demo_queries()

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import json
from mcp_server_msbuild import MSBuildAnalyticsServer

async def test_queries():
    """Test the analytics queries directly"""
    
    print("Testing MSBuild Analytics Queries")
    print("=" * 50)
    
    server = MSBuildAnalyticsServer(r"c:\Users\ykovalova\msbuild\data")
    
    try:
        # Initialize
        await server._initialize_db()
        print("✓ Database initialized")
        
        # Test get_statistics
        print("\nTesting get_statistics...")
        stats = await server._get_statistics(include_trends=True)
        print(f"✓ Statistics retrieved: {len(stats)} top-level keys")
        print(f"  - Total issues: {stats.get('issues', {}).get('total', 0)}")
        print(f"  - Total PRs: {stats.get('pull_requests', {}).get('total', 0)}")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if server.conn:
            server.conn.close()

if __name__ == "__main__":
    asyncio.run(test_queries())
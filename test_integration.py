#!/usr/bin/env python3
"""
Integration Test Script for The Seer v1 Financial Analysis Agent

This script tests the complete workflow from API request to analysis report
using sample Indian stock tickers.

Requirements:
- All environment variables must be set (API keys for Twelve Data, Serper, etc.)
- Redis and other dependencies must be running
- FastAPI server should be running on localhost:8080

Usage:
    python test_integration.py
"""

import asyncio
import time
import json
import httpx
from typing import Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from common.schemas.entities import AnalysisRequest, AnalysisType, AnalysisTimeHorizon


# Test configuration
BASE_URL = "http://localhost:8080"
TEST_TICKERS = [
    "RELIANCE.NS",  # Oil & Gas giant
    "TCS.NS",       # IT services leader
    "INFY.NS",      # IT services
]

class IntegrationTester:
    """Integration test suite for financial analysis agent"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=600.0)  # 10 minute timeout
        self.test_results = []
    
    async def test_health_check(self) -> bool:
        """Test basic service health"""
        print("🔍 Testing service health...")
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Service health check passed: {data.get('message', 'OK')}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    async def test_analysis_endpoints_health(self) -> bool:
        """Test analysis service health"""
        print("🔍 Testing analysis service health...")
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/analysis/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Analysis service health check passed: {data.get('service', 'OK')}")
                return True
            else:
                print(f"❌ Analysis health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Analysis health check error: {str(e)}")
            return False
    
    async def test_single_stock_analysis(self, ticker: str) -> Dict[str, Any]:
        """Test complete analysis workflow for a single stock"""
        print(f"\n📊 Testing analysis for {ticker}...")
        
        test_result = {
            "ticker": ticker,
            "success": False,
            "session_id": None,
            "duration": 0,
            "errors": [],
            "analysis_quality": {}
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Start analysis
            print(f"  🚀 Starting analysis for {ticker}...")
            
            analysis_request = {
                "ticker": ticker,
                "analysis_type": "comprehensive",
                "time_horizon": "medium_term",
                "include_technical": True,
                "include_fundamental": True,
                "include_news": True,
                "include_sector": True
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/analysis/start",
                json=analysis_request
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to start analysis: {response.status_code}"
                test_result["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
                return test_result
            
            data = response.json()
            session_id = data["session_id"]
            test_result["session_id"] = session_id
            print(f"  ✅ Analysis started with session ID: {session_id}")
            
            # Step 2: Monitor progress
            print(f"  ⏳ Monitoring analysis progress...")
            max_wait_time = 300  # 5 minutes
            check_interval = 10   # 10 seconds
            checks = 0
            
            while checks * check_interval < max_wait_time:
                await asyncio.sleep(check_interval)
                checks += 1
                
                # Check status
                status_response = await self.client.get(
                    f"{self.base_url}/api/v1/analysis/status/{session_id}"
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    progress = status_data.get("progress_percentage", 0)
                    current_step = status_data.get("current_step", "unknown")
                    status = status_data.get("status", "unknown")
                    
                    print(f"    📈 Progress: {progress:.1f}% - {current_step}")
                    
                    if status == "completed":
                        print(f"  ✅ Analysis completed!")
                        break
                    elif status == "failed":
                        error_msg = status_data.get("error_message", "Analysis failed")
                        test_result["errors"].append(f"Analysis failed: {error_msg}")
                        print(f"  ❌ Analysis failed: {error_msg}")
                        return test_result
                else:
                    print(f"    ⚠️ Status check failed: {status_response.status_code}")
            
            # Step 3: Get results
            print(f"  📋 Retrieving analysis results...")
            
            result_response = await self.client.get(
                f"{self.base_url}/api/v1/analysis/result/{session_id}"
            )
            
            if result_response.status_code == 200:
                result_data = result_response.json()
                
                if result_data.get("status") == "completed" and result_data.get("analysis"):
                    print(f"  ✅ Analysis results retrieved successfully!")
                    
                    # Analyze result quality
                    analysis = result_data["analysis"]
                    quality_metrics = self._assess_analysis_quality(analysis)
                    test_result["analysis_quality"] = quality_metrics
                    
                    test_result["success"] = True
                    
                    # Print summary
                    print(f"  📊 Analysis Summary for {ticker}:")
                    print(f"    • Executive Summary: {len(analysis.get('executive_summary', ''))} chars")
                    print(f"    • Investment Recommendation: {analysis.get('investment_recommendation', 'N/A')}")
                    
                    confidence_scores = analysis.get('confidence_scores', {})
                    if confidence_scores:
                        print(f"    • Overall Confidence: {confidence_scores.get('overall', 'N/A')}/10")
                    
                    price_targets = analysis.get('price_targets', {})
                    if price_targets:
                        current_price = price_targets.get('current_price', 0)
                        base_target = price_targets.get('base_case_target', 0)
                        if current_price and base_target:
                            upside = ((base_target - current_price) / current_price) * 100
                            print(f"    • Current Price: ₹{current_price:.2f}")
                            print(f"    • Base Case Target: ₹{base_target:.2f} ({upside:+.1f}%)")
                    
                else:
                    error_msg = f"Analysis not completed or missing data: {result_data.get('status', 'unknown')}"
                    test_result["errors"].append(error_msg)
                    print(f"  ❌ {error_msg}")
            else:
                error_msg = f"Failed to get results: {result_response.status_code}"
                test_result["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
            
        except Exception as e:
            error_msg = f"Test error: {str(e)}"
            test_result["errors"].append(error_msg)
            print(f"  ❌ {error_msg}")
        
        finally:
            test_result["duration"] = time.time() - start_time
            print(f"  ⏱️ Total duration: {test_result['duration']:.1f} seconds")
        
        return test_result
    
    def _assess_analysis_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of analysis output"""
        quality = {
            "completeness_score": 0,
            "content_quality_score": 0,
            "structure_score": 0,
            "overall_score": 0
        }
        
        # Check completeness
        required_sections = [
            "executive_summary", "market_context", "fundamental_analysis",
            "technical_analysis", "news_sentiment", "risk_assessment",
            "price_targets", "investment_recommendation"
        ]
        
        present_sections = sum(1 for section in required_sections if analysis.get(section))
        quality["completeness_score"] = (present_sections / len(required_sections)) * 100
        
        # Check content quality (basic heuristics)
        executive_summary = analysis.get("executive_summary", "")
        quality["content_quality_score"] = min(len(executive_summary) / 200 * 100, 100)  # Expect at least 200 chars
        
        # Check structure (confidence scores presence)
        confidence_scores = analysis.get("confidence_scores", {})
        quality["structure_score"] = min(len(confidence_scores) / 4 * 100, 100)  # Expect 4 confidence scores
        
        # Overall score
        quality["overall_score"] = (
            quality["completeness_score"] * 0.5 +
            quality["content_quality_score"] * 0.3 +
            quality["structure_score"] * 0.2
        )
        
        return quality
    
    async def test_batch_analysis(self) -> Dict[str, Any]:
        """Test batch analysis functionality"""
        print(f"\n🔄 Testing batch analysis...")
        
        try:
            batch_request = {
                "tickers": TEST_TICKERS[:2],  # Test with 2 tickers
                "analysis_type": "quick",
                "time_horizon": "medium_term"
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/analysis/batch",
                json=batch_request
            )
            
            if response.status_code == 200:
                data = response.json()
                success_count = data.get("success_count", 0)
                error_count = data.get("error_count", 0)
                
                print(f"  ✅ Batch analysis started:")
                print(f"    • Successful: {success_count}")
                print(f"    • Errors: {error_count}")
                
                return {
                    "success": True,
                    "session_ids": data.get("session_ids", {}),
                    "errors": data.get("errors", {})
                }
            else:
                print(f"  ❌ Batch analysis failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ Batch analysis error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        print("🚀 Starting comprehensive integration tests for The Seer v1...")
        print("=" * 60)
        
        overall_results = {
            "timestamp": time.time(),
            "tests": {},
            "summary": {}
        }
        
        # Test 1: Health checks
        health_ok = await self.test_health_check()
        analysis_health_ok = await self.test_analysis_endpoints_health()
        overall_results["tests"]["health_checks"] = {
            "main_service": health_ok,
            "analysis_service": analysis_health_ok
        }
        
        if not health_ok or not analysis_health_ok:
            print("\n❌ Health checks failed. Skipping analysis tests.")
            return overall_results
        
        # Test 2: Single stock analyses
        print(f"\n🧪 Running single stock analysis tests...")
        analysis_results = []
        
        for ticker in TEST_TICKERS[:2]:  # Test first 2 tickers
            result = await self.test_single_stock_analysis(ticker)
            analysis_results.append(result)
            self.test_results.append(result)
        
        overall_results["tests"]["single_analysis"] = analysis_results
        
        # Test 3: Batch analysis
        batch_result = await self.test_batch_analysis()
        overall_results["tests"]["batch_analysis"] = batch_result
        
        # Summary
        successful_analyses = sum(1 for r in analysis_results if r["success"])
        overall_results["summary"] = {
            "total_tests": len(analysis_results) + 1,  # +1 for batch
            "successful_analyses": successful_analyses,
            "average_duration": sum(r["duration"] for r in analysis_results) / len(analysis_results),
            "overall_success": successful_analyses >= len(analysis_results) // 2  # At least 50% success
        }
        
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Successful analyses: {successful_analyses}/{len(analysis_results)}")
        print(f"⏱️ Average duration: {overall_results['summary']['average_duration']:.1f}s")
        print(f"🎯 Overall success: {'PASS' if overall_results['summary']['overall_success'] else 'FAIL'}")
        
        return overall_results
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Main test runner"""
    tester = IntegrationTester()
    
    try:
        results = await tester.run_comprehensive_test()
        
        # Save results
        with open("integration_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to integration_test_results.json")
        
        # Exit with appropriate code
        if results["summary"]["overall_success"]:
            print("🎉 Integration tests PASSED!")
            sys.exit(0)
        else:
            print("💥 Integration tests FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {str(e)}")
        sys.exit(1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # Check if server is likely running
    print("🔧 Pre-flight checks...")
    
    # Basic environment check
    required_env_vars = [
        "TWELVE_DATA_API_KEY",
        "SERPER_API_KEY", 
        "GEMINI_API_KEY",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your .env file before running tests.")
        sys.exit(1)
    
    print("✅ Environment variables configured")
    print("⚠️ Make sure the FastAPI server is running on http://localhost:8080")
    print("⚠️ Make sure Redis and other dependencies are available")
    
    # Run tests
    asyncio.run(main()) 
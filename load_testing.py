import argparse
import json
import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

def send_request(url: str, prompt: str, request_id: int, max_tokens: int = 128, temperature: float = 0.7) -> Dict[str, Any]:
    """发送单个请求到LLM服务器"""
    payload = {
        "model": "Qwen3-1.7B-Q8_0",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            data = response.json()
            latency = time.perf_counter() - start_time
            content = data["choices"][0]["message"]["content"]
            tokens = len(content.split())  # 简单分词估算token数
            return {
                "success": True,
                "latency": latency,
                "tokens": tokens,
                "request_id": request_id
            }
        else:
            return {
                "success": False,
                "status": response.status_code,
                "error": response.text[:200] if response.text else "",
                "request_id": request_id
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "request_id": request_id
        }

def parallel_execute_requests(
    url: str,
    prompt: str,
    total_requests: int,
    max_workers: int = 10,
    jitter: float = 0.0
) -> List[Dict[str, Any]]:
    """并行执行多个请求"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(total_requests):
            # 创建任务并添加jitter
            futures.append(executor.submit(send_request, url, prompt, i))
            if jitter > 0.0:
                time.sleep(jitter * random.random())
        
        # 按完成顺序收集结果
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 10 == 0 or i == total_requests - 1:
                print(f"已完成 {i+1}/{total_requests} 请求 | 成功率: {len([r for r in results if r['success']]) / (i+1):.1%}")
    
    return results

def generate_report(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """生成压测报告"""
    successful = [r for r in results if r["success"]]
    failed = len(results) - len(successful)
    
    report = {
        "config": config,
        "summary": {
            "total_requests": len(results),
            "concurrency": config["max_workers"],
            "jitter": config["jitter"],
            "success_rate": len(successful) / len(results) if results else 0,
            "failed": failed
        }
    }
    
    if successful:
        latencies = [r["latency"] for r in successful]
        tokens_list = [r["tokens"] for r in successful]
        
        report["summary"]["avg_latency"] = sum(latencies) / len(latencies)
        report["summary"]["min_latency"] = min(latencies)
        report["summary"]["max_latency"] = max(latencies)
        
        total_tokens = sum(tokens_list)
        report["summary"]["total_tokens"] = total_tokens
        report["summary"]["avg_tokens_per_request"] = total_tokens / len(successful)
        report["summary"]["tokens_per_second"] = total_tokens / (report["summary"]["avg_latency"] * config["max_workers"])
    
    # 错误分析
    if failed > 0:
        errors = {}
        for result in results:
            if not result["success"]:
                error_key = result.get("status", "exception") or result.get("error", "unknown")
                errors.setdefault(str(error_key), 0)
                errors[str(error_key)] += 1
        report["error_analysis"] = errors
    
    return report

def main():
    parser = argparse.ArgumentParser(description="LLM服务器压力测试工具")
    parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions", 
                        help="API端点URL")
    parser.add_argument("--prompt", default="解释量子力学的基本原理", 
                        help="测试提示词")
    parser.add_argument("--requests", type=int, default=100, 
                        help="总请求数")
    parser.add_argument("--concurrency", type=int, default=10, 
                        help="并发工作线程数")
    parser.add_argument("--jitter", type=float, default=0.0, 
                        help="请求间隔抖动系数 (0.0-1.0)")
    parser.add_argument("--output", default="stress_test_report.json", 
                        help="输出报告文件名")
    args = parser.parse_args()
    
    print(f"开始压力测试: {args.requests} 请求, {args.concurrency} 并发, jitter={args.jitter}")
    print(f"提示词: '{args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}'")
    
    start_time = time.perf_counter()
    
    # 执行压测
    results = parallel_execute_requests(
        url=args.url,
        prompt=args.prompt,
        total_requests=args.requests,
        max_workers=args.concurrency,
        jitter=args.jitter
    )
    
    total_time = time.perf_counter() - start_time
    
    # 生成报告
    config = {
        "url": args.url,
        "prompt": args.prompt,
        "total_requests": args.requests,
        "max_workers": args.concurrency,
        "jitter": args.jitter
    }
    report = generate_report(results, config)
    report["summary"]["total_time"] = total_time
    report["summary"]["throughput"] = args.requests / total_time
    
    # 打印摘要
    print("\n===== 压测结果摘要 =====")
    print(f"总请求: {report['summary']['total_requests']}")
    print(f"并发数: {report['summary']['concurrency']}")
    print(f"成功率: {report['summary']['success_rate']:.2%}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"吞吐量: {report['summary']['throughput']:.1f} 请求/秒")
    
    if 'avg_latency' in report['summary']:
        print(f"平均延迟: {report['summary']['avg_latency']:.3f}秒")
        print(f"最小延迟: {report['summary']['min_latency']:.3f}秒")
        print(f"最大延迟: {report['summary']['max_latency']:.3f}秒")
        print(f"Token速率: {report['summary']['tokens_per_second']:.1f} tokens/秒")
    
    # 保存完整报告
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整报告已保存至: {args.output}")

if __name__ == "__main__":
    main()

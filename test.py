import asyncio
import time

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import MemoryAdaptiveDispatcher  # or SemaphoreDispatcher


# Edit this list to match the URLs you want to benchmark
URLS = [
    "https://marleycoffee.com/marleycoffee-com-contest-rules-regs-holiday-gift",
    "https://marleycoffee.com/marleycoffee-com-contest-rules-regs-back-to-school-2023/",
    "https://mother-parkers.com/driving-growth-with-private-label-coffee-cans/",
    "https://mother-parkers.com/driving-coffee-in-convenience/",
    "https://marleycoffee.com/social_5/",
]


async def main():
    # One config for all URLs (non-streaming, bypass cache for clean timing)
    run_config = CrawlerRunConfig(
        stream=False,
        cache_mode=CacheMode.BYPASS,
        page_timeout=12_000
    )

    # Optional: custom dispatcher to control concurrency
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,  # start throttling if memory is high
        max_session_permit=10,          # max concurrent browser sessions
    )

    async with AsyncWebCrawler() as crawler:
        start_time = time.perf_counter()

        results = await crawler.arun_many(
            urls=URLS,
            config=run_config,
            dispatcher=dispatcher,  # you can pass None to use default
        )

        elapsed = time.perf_counter() - start_time

    # Basic stats
    total = len(results)
    success = sum(1 for r in results if r.success)
    failed = total - success
    rps = total / elapsed if elapsed > 0 else 0.0

    print(f"Total URLs     : {total}")
    print(f"Success        : {success}")
    print(f"Failed         : {failed}")
    print(f"Elapsed time   : {elapsed:.3f} s")
    print(f"Throughput     : {rps:.2f} req/s")
    print()

    # Optional: print per-URL timing if available in dispatch_result
    for r in results:
        dur = None
        if getattr(r, "dispatch_result", None):
            dr = r.dispatch_result
            if getattr(dr, "start_time", None) and getattr(dr, "end_time", None):
                dur = dr.end_time - dr.start_time

        status = "OK" if r.success else f"FAIL ({r.error_message})"
        if dur is not None:
            print(f"{r.url} -> {status}, duration={dur:.3f}s")
        else:
            print(f"{r.url} -> {status}")


if __name__ == "__main__":
    asyncio.run(main())

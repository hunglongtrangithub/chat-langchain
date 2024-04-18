import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

def test_call():
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(requests.get, "http://localhost:8080/task") for _ in range(5)]
        for future in futures:
            resp = future.result()
            print(resp.text)

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def test_async_call():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, "http://localhost:8080/async_task") for _ in range(5)]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result)

if __name__ == "__main__":
    # Uncomment the function you want to run
    asyncio.run(test_async_call())  # For async testing
    # test_call()  # For sync testing

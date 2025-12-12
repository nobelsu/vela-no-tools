from utils import read_text, write_text, row_to_formatted_string
import time 
import asyncio 
import subprocess
import improver as Improver
import csv

async def train(startt):
    start = time.time()
    with open("data/test.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = []
        actual = []
        counter = 0
        for row in reader:
            counter += 1
            write_text(counter)
            if counter < startt:
                continue
            rows.append(row_to_formatted_string(row))
            actual.append(row.get("success"))
            if len(rows) >= 10:
                subprocess.run(["uv", "run", "predictor.py", "--p"] + rows + ["--a"] + actual, capture_output=True, text=True)
                instructions = read_text("report.txt")
                write_text("instructions.txt", await Improver.runImprover(instructions))
                rows = []
                actual = []
    end = time.time()
    t = end - start
    print(f"\n\nTotal run time: {t:.2f}s")

if __name__ == "__main__":
    asyncio.run(train(0))

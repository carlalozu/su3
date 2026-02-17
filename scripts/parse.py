import re
import csv
import sys

pattern = re.compile(
    r'(?P<layout>\w+)\s+'
    r'(?P<phase>\w+)\s+'
    r'total=(?P<total>[0-9.]+)\s+s\s+\|\s+'
    r'avg=(?P<avg>[0-9.]+)\s+s\s+\|\s+'
    r'n=(?P<n>\d+)\s+\|\s+'
    r'vol=(?P<vol>\d+)\s+\|\s+'
    r'cache=(?P<cache>\d+)\s+\|\s+'
    r'threads=(?P<threads>\d+)'
)

writer = csv.writer(sys.stdout)
writer.writerow([
    "layout", "phase", "total_s", "avg_s",
    "n", "vol", "cache", "threads"
])

for line in sys.stdin:
    match = pattern.search(line)
    if match:
        writer.writerow([
            match["layout"],
            match["phase"],
            match["total"],
            match["avg"],
            match["n"],
            match["vol"],
            match["cache"],
            match["threads"],
        ])

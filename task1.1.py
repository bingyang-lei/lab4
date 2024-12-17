# 使用spark RDD编程，完成以下任务：
# 查询每一天日期的总资金流入和流出情况： 使用 user_balance_table ，
# 计算出所有用户在每一天的总资金流入和总资金流出量。
from pyspark import SparkContext, SparkConf

# 初始化SparkContext
conf = SparkConf().setAppName("DailyFlow").setMaster("local")
sc = SparkContext(conf=conf)

# 读取数据
data = sc.textFile("./Purchase/user_balance_table.csv")

# 解析数据，过滤掉表头
header = data.first()
data = data.filter(lambda line: line != header)

# 解析每一行数据
def parse_line(line):
    fields = line.split(',')
    date = fields[1]
    total_purchase_amt = float(fields[4]) if fields[4] else 0.0
    total_redeem_amt = float(fields[8]) if fields[8] else 0.0
    return (date, (total_purchase_amt, total_redeem_amt))

parsed_data = data.map(parse_line)

# 按日期聚合数据
aggregated_data = parsed_data.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# 格式化输出
formatted_data = aggregated_data.map(lambda x: f"{x[0]} {int(x[1][0])} {int(x[1][1])}")

# 收集结果到驱动程序
result = formatted_data.collect()

# 将结果写入单个文件
with open("./output/task1.1-result.txt", "w") as f:
    for line in result:
        f.write(line + "\n")

# 停止SparkContext
sc.stop()
"""
活跃用户分析： 使用 user_balance_table ，定义活跃用户为在指定月份内有至少5天记录的
用户，统计2014年8月的活跃用户总数。
"""
from pyspark import SparkContext, SparkConf

# 初始化SparkContext
conf = SparkConf().setAppName("ActiveUsers").setMaster("local")
sc = SparkContext(conf=conf)

# 读取数据
data = sc.textFile("./Purchase/user_balance_table.csv")

# 解析数据，过滤掉表头
header = data.first()
data = data.filter(lambda line: line != header)

# 解析每一行数据
def parse_line(line):
    fields = line.split(',')
    user_id = fields[0]
    date = fields[1]
    return (user_id, date)

parsed_data = data.map(parse_line)

# 过滤出2014年8月的数据
august_data = parsed_data.filter(lambda x: x[1].startswith('201408'))

# 统计每个用户在2014年8月的记录天数
user_days = august_data.distinct().map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

# 过滤出活跃用户（记录天数 >= 5）
active_users = user_days.filter(lambda x: x[1] >= 5)

# 统计活跃用户总数
active_user_count = active_users.count()

# 输出结果
print(active_user_count)

# 停止SparkContext
sc.stop()
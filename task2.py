from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum

# 初始化SparkSession
spark = SparkSession.builder.appName("CityAverageBalanceAndTopUsers").getOrCreate()

# 读取数据
user_balance_df = spark.read.csv("./Purchase/user_balance_table.csv", header=True, inferSchema=True)
user_profile_df = spark.read.csv("./Purchase/user_profile_table.csv", header=True, inferSchema=True)

# 任务1：计算每个城市在2014年3月1日的用户平均余额( tBalance )，按平均余额降序排列
# 过滤出2014年3月1日的数据
march_first_data = user_balance_df.filter(col("report_date") == "20140301")

# 关联user_profile_table获取城市信息
march_first_data_with_city = march_first_data.join(user_profile_df, "user_id")

# 计算每个城市的用户平均余额
average_balance = march_first_data_with_city.groupBy("City").agg({"tBalance": "avg"}).withColumnRenamed("avg(tBalance)", "average_balance")

# 按平均余额降序排列
sorted_average_balance = average_balance.orderBy(col("average_balance").desc())

# 显示结果
sorted_average_balance.show()

# 任务2：统计每个城市总流量前3高的用户
# 过滤出2014年8月的数据
august_data = user_balance_df.filter(col("report_date").startswith("201408"))

# 计算每个用户在2014年8月的总流量
august_data_with_flow = august_data.withColumn("total_flow", col("total_purchase_amt") + col("total_redeem_amt"))

# 关联user_profile_table获取城市信息
august_data_with_city = august_data_with_flow.join(user_profile_df, "user_id")

# 计算每个城市中每个用户的总流量
user_total_flow = august_data_with_city.groupBy("City", "user_id").agg(_sum("total_flow").alias("total_flow"))

# 获取每个城市总流量前3高的用户
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window_spec = Window.partitionBy("City").orderBy(col("total_flow").desc())
ranked_users = user_total_flow.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 3)

# 显示结果
ranked_users.select("City", "user_id", "total_flow").show(ranked_users.count())

# 停止SparkSession
spark.stop()
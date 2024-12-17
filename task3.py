from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, round
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, LongType

# 初始化SparkContext和SparkSession
conf = SparkConf().setAppName("PredictPurchaseRedeem").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# 读取数据
data = spark.read.csv("./Purchase/user_balance_table.csv", header=True, inferSchema=True)

# 过滤出2014年8月之前的数据
data = data.filter((col("report_date") > '20140730') & (col("report_date") < '20140901'))
# 打印data的数量
print(data.count())

# 按日期聚合数据，计算每天的总申购和总赎回
daily_data = data.groupBy("report_date").agg(
    _sum("total_purchase_amt").alias("total_purchase_amt"),
    _sum("total_redeem_amt").alias("total_redeem_amt")
)

# 准备训练数据
assembler = VectorAssembler(inputCols=["report_date"], outputCol="features")
training_data = assembler.transform(daily_data)

# 训练申购总额的线性回归模型
lr_purchase = LinearRegression(featuresCol="features", labelCol="total_purchase_amt")
lr_purchase_model = lr_purchase.fit(training_data)

# 训练赎回总额的线性回归模型
lr_redeem = LinearRegression(featuresCol="features", labelCol="total_redeem_amt")
lr_redeem_model = lr_redeem.fit(training_data)

# 生成2014年9月的日期数据
schema = StructType([StructField("report_date", LongType(), True)])
september_dates = spark.createDataFrame([(20140901 + i,) for i in range(30)], schema)

# 准备预测数据
september_data = assembler.transform(september_dates)

# 预测申购总额
purchase_predictions = lr_purchase_model.transform(september_data).select("report_date", round(col("prediction"), 2).alias("purchase"))

# 预测赎回总额
redeem_predictions = lr_redeem_model.transform(september_data).select("report_date", round(col("prediction"), 2).alias("redeem"))

# 合并预测结果
predictions = purchase_predictions.join(redeem_predictions, "report_date")

# 将结果转换为字符串格式，保留两位小数，并避免科学计数法
formatted_predictions = predictions.select(
    col("report_date"),
    col("purchase").cast("decimal(20,2)").alias("purchase"),
    col("redeem").cast("decimal(20,2)").alias("redeem")
)

# 将结果保存到表格
formatted_predictions.write.csv("./output/tc_comp_predict_table", header=True)

# 停止SparkContext和SparkSession
sc.stop()
spark.stop()



# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, sum as _sum, round
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.feature import VectorAssembler
# from pyspark.sql.types import StructType, StructField, LongType, DoubleType

# # 初始化SparkContext和SparkSession
# conf = SparkConf().setAppName("PredictPurchaseRedeem").setMaster("local")
# sc = SparkContext(conf=conf)
# spark = SparkSession(sc)

# # 定义数据的schema
# schema = StructType([
#     StructField("report_date", LongType(), True),
#     StructField("purchase", DoubleType(), True),
#     StructField("redeem", DoubleType(), True)
# ])

# # 读取数据
# data = spark.read.csv("./Purchase/task1.1-result.csv", schema=schema, header=True, inferSchema=False)

# # 过滤出2014年7月和2014年8月的数据
# data = data.filter((col("report_date") >= 20140701) & (col("report_date") <= 20140831))

# # 准备训练数据
# assembler = VectorAssembler(inputCols=["report_date"], outputCol="features")
# training_data = assembler.transform(data)

# # 训练申购总额的线性回归模型
# lr_purchase = LinearRegression(featuresCol="features", labelCol="purchase")
# lr_purchase_model = lr_purchase.fit(training_data)

# # 训练赎回总额的线性回归模型
# lr_redeem = LinearRegression(featuresCol="features", labelCol="redeem")
# lr_redeem_model = lr_redeem.fit(training_data)

# # 生成2014年9月的日期数据
# schema = StructType([StructField("report_date", LongType(), True)])
# september_dates = spark.createDataFrame([(20140901 + i,) for i in range(30)], schema)

# # 准备预测数据
# september_data = assembler.transform(september_dates)

# # 预测申购总额
# purchase_predictions = lr_purchase_model.transform(september_data).select("report_date", round(col("prediction"), 2).alias("purchase"))

# # 预测赎回总额
# redeem_predictions = lr_redeem_model.transform(september_data).select("report_date", round(col("prediction"), 2).alias("redeem"))

# # 合并预测结果
# predictions = purchase_predictions.join(redeem_predictions, "report_date")

# # 将结果转换为字符串格式，保留两位小数，并避免科学计数法
# formatted_predictions = predictions.select(
#     col("report_date"),
#     col("purchase").cast("decimal(20,2)").alias("purchase"),
#     col("redeem").cast("decimal(20,2)").alias("redeem")
# )

# # 将结果保存到CSV文件
# formatted_predictions.write.csv("./output/tc_comp_predict_table2", header=True)

# # 停止SparkContext和SparkSession
# sc.stop()
# spark.stop()
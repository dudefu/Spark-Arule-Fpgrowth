package main.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.SortedSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.mllib.fpm.AssociationRules
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.mllib.fpm.FPGrowthModel


object AssociationMain {

  def main(arg: Array[String]) {

    val conf = new SparkConf().setAppName("Spark FPGrowth").registerKryoClasses(Array(classOf[ArrayBuffer[String]], classOf[ListBuffer[String]]))
    val sparkContext: SparkContext = new SparkContext(conf)
    val filePath: String = arg(0)
    val model = FPGrowthTree1(filePath, sparkContext)
    val rule = Arule(model)
    println("Number of frequent itemsets: " + model.freqItemsets.count() + "\n")
    model.freqItemsets.collect().foreach { itemset => println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq) }
    rule.collect().foreach { rule => println("[" + rule.antecedent.mkString(",") + "=>" + rule.consequent.mkString(",") + "]," + rule.confidence)
    }
  }

  def FPGrowthTree1(filePath: String, sparkContext: SparkContext): FPGrowthModel[String] = {
    val transactions = sparkContext.textFile(filePath).map(x => x.split(" ")).cache()
    println("Number of transactions:" + transactions.count() + "\n")
    return new FPGrowth().setMinSupport(0.6).run(transactions)

  }

  def Arule(model: FPGrowthModel[String]): RDD[AssociationRules.Rule[String]] = {
    val ar = new AssociationRules().setMinConfidence(1)
    return ar.run(model.freqItemsets)

  }

}

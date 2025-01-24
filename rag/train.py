from rag.rag_agent import RAG
import dspy
from dspy import Example
import mlflow
from dotenv import load_dotenv
import os
from dspy.datasets import HotPotQA
from dspy.evaluate import SemanticF1

load_dotenv()

mlflow.dspy.autolog()
mlflow.set_experiment("RAG Training")

lm = dspy.LM("deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct")
dspy.configure(lm=lm)

dataset = HotPotQA(train_seed=1, train_size=15, eval_seed=2023, dev_size=50, test_size=50)
trainset = [Example(question=x.question, response=x.answer).with_inputs("question") for x in dataset.train]
devset = [Example(question=x.question, response=x.answer).with_inputs("question") for x in dataset.dev]
testset = [Example(question=x.question, response=x.answer).with_inputs("question") for x in dataset.test]

metric = SemanticF1(decompositional=True)
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2, provide_traceback=True)
optimizer = dspy.MIPROv2(
    metric=metric, auto="light", num_threads=24, max_bootstrapped_demos=4, max_labeled_demos=4
)

rag = optimizer.compile(RAG(), trainset=trainset, valset=testset)
# rag = RAG()
evaluate(rag)


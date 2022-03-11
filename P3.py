from CandidateEliminationAlgorithm import CandidateElimination
import csv

with open('./datasets/P3.csv')  as csvFile:
        data = [tuple(line) for line in csv.reader(csvFile)]
ce = CandidateElimination(data)
ce.candidate_elimination()
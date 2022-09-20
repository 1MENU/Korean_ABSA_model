import argparse
import json

def typeCheck(eachTask, eachTaskPred):
	# 각 데이터셋의 수량
	taskGoldLenDict = {"boolq":704, "copa":500, "cola":1060, "wic":1246}
	taskGoldStartIdxDict = {"boolq":1, "copa":1, "cola":0, "wic":1}
	taskGoldLabelDict = {"boolq":[0,1], "copa":[1,2], "cola":[0,1], "wic":[True, False]}

	for idx, eachPred in enumerate(eachTaskPred):
		# type check
		if type(eachPred["idx"])==int: pass
		else: 
			print ("[ERROR]: The value type of JSON label(idx) in the submitted \"{}\" task file.".format(eachTask)); 
			return False 

		if eachTask in ["boolq", "cola", "copa"]:
			if type(eachPred["label"])==int: pass
			else: 
				print ("[ERROR]: The value type of JSON label(label) in the submitted \"{}\" task file.".format(eachTask)); 
				return False
		else:
			if type(eachPred["label"])==bool: pass
			else: 
				print ("[ERROR]: The value type of JSON label(label) in the submitted \"{}\" task file.".format(eachTask)); 
				return False

		# number of contents
		if taskGoldLenDict.get(eachTask) != len(eachTaskPred):
			print ("[ERROR]: the number of keys in the submitted \"{}\" task file.".format(eachTask))
			return False

		# start index check
		if idx==0 and eachPred["idx"] != taskGoldStartIdxDict.get(eachTask): 
			print("[ERROR]: start index number mismatch in the submitted \"{}\" task file.".format(eachTask))
			return False

def predParsing(predFile):
	try:
		with open(predFile, "r") as predJsonFile:
			predJson = json.load(predJsonFile)
		for idx, eachTask in enumerate(predJson):
			taskJson = predJson[eachTask]
			if eachTask=="boolq": typeCheck(eachTask, taskJson)
			elif eachTask=="copa":  typeCheck(eachTask, taskJson)
			elif eachTask=="cola":  typeCheck(eachTask, taskJson)
			else: typeCheck(eachTask, taskJson)
	except ValueError:
		print("Format Error : \"{}\" is not readable file.".format(predFile))

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='국립국어원 리더보드 제출 성능 측정 코드\t> $ python format_check.py -p 제출_파일')
	parser.add_argument('-p', '--predFile', required=True, help='path of prediction File')
	args=parser.parse_args()

	resultFormatCheck = predParsing(args.predFile)

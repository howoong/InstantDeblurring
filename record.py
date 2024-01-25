
record = {}
# 데이터셋별로 딕셔너리 만들어서 정렬 & 평균 구하기
with open("./res_baseline.txt", "r") as f:
    while(1):
        tmp = []
        for i in range(15):
            try:
                line = f.readline()
            except StopIteration:
                exit(1)
            if line == "\n":
                break
            prev = line
            val = line.split(" ")[1].split("=")[1]
            tmp.append(eval(val))
        print(prev.split(" ")[0])
        result = ''.join(str(item) + " " for item in tmp)
        print(result)
        print("====")





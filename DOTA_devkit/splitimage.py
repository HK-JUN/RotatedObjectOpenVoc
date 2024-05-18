from ImgSplit_multi_process import splitbase

# 원본 DOTA 데이터셋 경로와 분할된 데이터셋을 저장할 경로
basepath = '/home/jhpark/my_dota/DOTA_devkit/dataset/val'
outpath = '/home/jhpark/my_dota/DOTA_devkit/splitted_data/val'

# 이미지 분할 설정
split = splitbase(basepath=basepath, outpath=outpath, gap=100, subsize=386, num_process=8,ext='.png')

# 이미지 분할 실행
split.splitdata(0.1)  # rate=1은 이미지의 크기를 변경하지 않음을 의미
print("finish")
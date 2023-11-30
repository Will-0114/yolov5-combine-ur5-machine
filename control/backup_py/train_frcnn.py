from pytorch_utils import frcnn_utils
"""
## trainingset up
## 0. copy pytorch_utils into your working directory
## 1. put all images in "./train_data/JPEGImages"
## 2. put all annontations in "./train_data/Annotations"
## 3. run train_frcnn() with your parmas: 
##   no_epoch = 100
#    batch_size = 2, 
#    train_data_path = "./train_data", 
#    test_data_path = "./train_data", 
#    load_model_name = "./model/frcnn.pth"
#    save_model_name ="./model/frcnn.pth",
#    evaluate_period = 5: 設定多少epoch評估一次，且儲存model
## Note: 執行後發現電腦會變慢，但又沒有發現有 leaking的現象
## Note2: 如果一開始學習很慢，可以調整 lr = 0.01
"""
def main():
    num_classes = 4
    batch_size  = 2
    frcnn_utils.retrain_frcnn(no_epoch = 300, batch_size = 2, train_data_path = "./train_data", test_data_path = "./train_data", 
               load_model_name = "./model/frcnn_PowerRed.pth", save_model_name ="./model/frcnn_PowerRed.pth",
                evaluate_period = 5 )
    frcnn_utils.retrain_frcnn(no_epoch = 300, batch_size = 2, train_data_path = "./train_data", test_data_path = "./train_data", 
               load_model_name = "./model/frcnn_PowerRed.pth", save_model_name ="./model/frcnn_PowerRed.pth",
                evaluate_period = 5 )
    # frcnn_utils.train_frcnn(no_epoch = 100, num_classes = num_classes, batch_size = batch_size, train_data_path = "./train_data", 
    #         test_data_path = "./train_data", save_model_name ="./model/frcnn_PowerRed.pth", evaluate_period = 5 )

if __name__ == "__main__":
    main()
import cv2, os


cascade_path = "./models/haarcascade_frontalface_default.xml"
assert os.path.isfile(cascade_path), "カスケード分類器の定義が存在しません"
cascade = cv2.CascadeClassifier(cascade_path)
capture = cv2.VideoCapture(0)


while True:
    ret, frame = capture.read()
    #frameはstr型らしい
    if ret == False:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_mosaic = frame_gray
    #グレースケールの一部をモザイクに用いる

    front_face_list = cascade.detectMultiScale(frame_mosaic)
    for (x, y, width, height) in front_face_list:
        frame_mosaic = cv2.rectangle(frame, (x, y), (x+width, y+height), color=(0, 0, 0), thickness=1)
        #モザイクの範囲をとる矩形を貼り付けたframeをframe_mosaicに代入することでカラー映像化に成功
        #frame_mosaicをframeに変えるとグレースケールに変化

        first_make_mosaic = frame_mosaic[y:y+height, x:x+width]
        second_make_mosaic = cv2.resize(first_make_mosaic, (width//25, height//25))
        #25が実用に耐えうるレベル
        third_make_mosaic = cv2.resize(second_make_mosaic, (width, height))
        frame_mosaic[y:y+height, x:x+width] = third_make_mosaic
        """
        frame_mosaicにモザイクの情報を追加しているのでは？
        """

    cv2.imshow("Video Mosaic Face", frame_mosaic)
    cv2.waitKey(1)


capture.release()
cv2.destroyAllWindows()
#OpenCVは例外を送出しないようにできているらしい

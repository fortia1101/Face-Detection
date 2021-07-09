import cv2, os, time


cascade_path = "./models/haarcascade_frontalface_default.xml"
#学習済みモデルの導入
#haarcascade_frontalface_alt_tree.xmlだとカメラ映像が表示されなかった
assert os.path.isfile(cascade_path), 'カスケード分類器の定義ファイルがありません'
#os.path.isfile()がFalseの場合に機能する
cascade = cv2.CascadeClassifier(cascade_path)
#カスケード分類器の特徴量を取得

capture = cv2.VideoCapture(1)
width = 1000
height = 600


count_start = time.perf_counter()
while True:
    ret, frame = capture.read()

    frame = cv2.resize(frame, (width, height))
    capture_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #高速で読み取れるようにグレースケールに変換
    front_face_list = cascade.detectMultiScale(capture_gray, minNeighbors=20)
    #minSize：物体がとりうる最小サイズで、この値より小さい物体は無視される
    #scaleFactor：画像スケールの縮小量
    #minNeighbors：物体となる矩形が含む必要のある、必要最低限の近傍矩形数
    """
    front_face_listは空のまま←これが正常?
    """

    for (x, y, w, h) in front_face_list:
        #x,y：顔の座標
        #w：顔の幅, h：顔の縦の長さ
        color = (0, 0, 250)
        #赤
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)
        #検出した顔を囲む長方形の生成

    count_stop = time.perf_counter()
    text = "DetectingTime: {0:.2f}sec" .format(count_stop-count_start)
    cv2.putText(frame, text=text, org=(130, 595), color=(250, 250, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
    cv2.imshow("frame_detected", frame)
    #この行をfor文内に入れるとframeが零行列で返ってくる
    cv2.waitKey(1)
    #この一文はfor文 or while文に絶対必要


capture.release()
#release software and hardware resources
cv2.destroyAllWindows()

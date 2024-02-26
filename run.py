from helper import *
import logging
import json
import argparse
import time
from imutils.video import VideoStream
from imutils.video import FPS
import dlib
import threading
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject


logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
log = logging.getLogger(__name__)

with open("utils/config.json", "r") as file:
    config = json.load(file)


def parse_arguments():
	# function to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections"),
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args


detModeln = "person-detection-retail-0013"
reidModeln = "person-reidentification-retail-0277"

## Loading Detection Model
detModelPath = f"intel/{detModeln}/FP16/{detModeln}.xml"
detModel, detCompiled, detShape, detInput = load_model(xml_path=detModelPath, device=config["Device"])

## Loading Reid Model
reidModelPath = f"intel/{reidModeln}/FP16/{reidModeln}.xml"
reidModel, reidCompiled, reidShape, reidInput = load_model(xml_path=reidModelPath, device=config["Device"])

## global variable for storing FV and ID
global all_pos_id
all_pos_id = []

## main func
def detection():
    ## pass arguments function
    args = parse_arguments()

    ## loading model for inference
    inferDet = detCompiled.create_infer_request()
    inferreid = reidCompiled.create_infer_request()

    ## Reading a video or url
    if not args.get("input", False):
        log.info("- Starting the live stream")
        vs = VideoStream(config["url"]).start()
        time.sleep(0.2)
    else:
        log.info("- Starting the video")
        vs = cv2.VideoCapture(args["input"])

    if config["Thread"]:
        vs = threading.ThreadingClass(config["url"])


    height = None
    width = None

    ## defining centroid tracker
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    ## declearing required variables
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # initialize empty lists to store the counting data
    total = []
    move_out = []
    move_in =[]

    fps = FPS().start()

    while True:
        ## Reading frames
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            pass

        if args['input'] is not None and frame is None:
            break
        
        ## checking for img features
        height, width, _ = frame.shape

        status = "Waiting"
        rects = []

        ## Detection model starts
        if totalFrames % args['skip_frames'] == 0:
            status = "Detection"
            trackers = []
            resized_img = preprocessing(frame, detShape)
            inferDet.set_tensor(detInput, openrun.Tensor(resized_img))
            inferDet.infer()
            detResult = inferDet.get_output_tensor(0).data
            detections = detResult.reshape(-1, 7)
            for _, detection in enumerate(detections):
                                # xmin,   ymin,   xmax, ymax
                _, _, confidence, startX, startY, endX, endY = detection
                if confidence > args["confidence"]:  ## checking CF score
                    startX = int(max((startX * frame.shape[1]), 10))
                    startY = int(max((startY * frame.shape[0]), 10))
                    endX = int(min((endX * frame.shape[1]), frame.shape[1] - 10))
                    endY = int(min((endY * frame.shape[0]), frame.shape[0] - 10))

                    ## reid model starts
                    if status == "Detection":
                        person = frame[startY - 10:endY + 10,startX - 10:endX + 10, :]
                        resized_person = preprocessing(person, reidShape)
                        inferreid.set_tensor(reidInput, openrun.Tensor(resized_person))
                        inferreid.infer()
                        reidResult = inferreid.get_output_tensor(0).data
                        fin_res = np.expand_dims(np.asarray(list(reidResult)), axis = 0)[0]
                        
                    ## tracker updatation
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

        else:
            
            ## Tracking
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, height // 2), (width, height // 2), (0, 0, 0), 2)
        objects = ct.update(rects, False)

        for (objectID, centroid) in objects.items():

            ## updation for reid  
            if status == "Detection" and (status != "Tracking" or status != "Waiting"):
                if all_pos_id is not None:
                    matchID, max_sim = find_most_similar_pair(fin_res.flatten(), all_pos_id)
                    # print("Similarity1", matchID, max_sim)
                    if round(max_sim, 2) > 0.9:
                        objectID = matchID
                        ct.register(centroid, True, objectID)  
                all_pos_id.append((fin_res, objectID))
                # log.info(f"Re-Identify with ID {matchID, objectID, max_sim}")

            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                ## continue with tracking
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    ## upward tracking
                    if direction < 0 and centroid[1] < height // 2:
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True

                    ## downward tracking
                    elif direction > 0 and centroid[1] > height // 2:
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True
                        total = []
                        total.append(len(move_in) - len(move_out))

            ## Updating object in dict
            trackableObjects[objectID] = to

            ## Displaying tracked id in the screen
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        # construct a tuple of information we will be displaying on the frame
        info_status = [
        ("Exit", totalUp),
        ("Enter", totalDown),
        ("Status", status),
        ]
        info_total = [
        ("Total people inside", ', '.join(map(str, total))),
        ]

        # display the output
        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, height - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
            ## cam - (50, height - ((i * 50) + 1300)) | gen - (10, H - ((i * 20) + 20))

        for (i, (k, v)) in enumerate(info_total):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, height - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
            ## cam - (50, height - ((i*20) + 1250)) | gen - (265, height - ((i * 20) + 60))        

        cv2.imshow("Real-Time Monitoring", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print(f"Elapsed time: {round(fps.elapsed(), 2)}\nApprox. FPS: {round(fps.fps(), 2)}\nTotal UP: {totalUp}\nTotal DOWN: {totalDown}\nTotal Count: {total}")
    vs.release()
    cv2.destroyAllWindows()


## Running main
detection()



    
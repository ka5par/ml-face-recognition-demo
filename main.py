# This code is based on the tutorial and code on this articles: 
# https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340
# Some whistles and bells were added for the purpose of the demo 
# 
import face_recognition
import cv2
import numpy as np
import os
import glob
import copy
import dlib

# TODO: automatically choose the parameters of the locator and encoder based on the GPU availability!
if not dlib.DLIB_USE_CUDA:
    print("Warning: code is not running with GPU support!")

global faces_encodings
faces_encodings = []
global faces_names
faces_names = []

def load_faces():
    print("loading faces..")
    # TODO: only load the new faces
    global faces_encodings
    faces_encodings = []
    global faces_names
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'data/faces/')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()

    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)], model="large", num_jitters=10)[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
        # Create array of known names
        names[i] = names[i].replace(cur_direc, "")
        faces_names.append(names[i].split('/')[-1].split('.')[0])

    # return faces_encodings, faces_names

load_faces()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frame_step = 0
display_warning = 0
display_warning_delete = 0
display_noface_warning = 0

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    original_frame = copy.deepcopy(frame)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
    #if frame_step % 2 == 0:
        face_locations = face_recognition.face_locations( rgb_small_frame, model="cnn", number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations, model='large', num_jitters=10)
        face_names = []
        face_confidences = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding,) #  tolerance=0.6
            name = "Unknown"
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            confidence = None
            if matches[best_match_index]:
                name = faces_names[best_match_index]
                dists = np.array(1 - face_distances)
                normalized_dists = dists/np.sum(dists)
                # TODO: the calculated confidence based on current distances is not very good. try to come up with something better!
                confidence = normalized_dists[best_match_index] # 1-face_distances[best_match_index]
                # sig = 1 #np.std(dists)
                # print("sig: ", sig)
                # var = sig**2
                # d = np.exp(-dists**2)
                # print("d: ", d)
                # confidence_gaussian = (d/var) / np.sum(d/var)
                # print("cg:", confidence_gaussian)
                # print("cg sum: ", confidence_gaussian.sum())
                # confidence = confidence_gaussian[best_match_index]
            face_names.append(name)
            face_confidences.append(confidence)
    process_this_frame = not process_this_frame
    #frame_step += 1

    # Display the results
    for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidences):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Input text label with a name below the face
        cv2.rectangle(frame, (left, bottom - 24), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        conf_val_str = f' {conf*100:2.1f}%' if conf is not None else ''
        cv2.putText(frame, name + conf_val_str, (left + 3, bottom - 8), font, 0.5, (255, 255, 255), 1)

    # TODO: create a function for all of these 
    if face_names != [] and (np.array(face_names) == 'Unknown').any():
        font = cv2.FONT_HERSHEY_SIMPLEX
        # TODO: set the coordinates according to available space
        left_top = (10,10)
        right_bottom = (500,30)
        cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "Press 's' to save your face", (left_top[0], right_bottom[1]-5), font, 0.5, (255,255,255), 1)

    if face_names != [] and (np.array(face_names) != 'Unknown').any():
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_top = (10,30)
        right_bottom = (500,50)
        cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "Press 'd' to delete your face", (left_top[0], right_bottom[1]-5), font, 0.5, (255,255,255), 1)
 
    if display_warning:
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_top = (10,50)
        right_bottom = (500,70)
        cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "Too many faces or already recognized faces!", (left_top[0], right_bottom[1]-5), font, 0.5, (255,255,255), 1)
        display_warning -= 1

    if display_warning_delete:
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_top = (10,50)
        right_bottom = (500,70)
        cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "Too many faces or already non-existing faces!", (left_top[0], right_bottom[1]-5), font, 0.5, (255,255,255), 1)
        display_warning_delete -= 1

    if display_noface_warning:
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_top = (10,35)
        right_bottom = (500,55)
        cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "No face detected!", (left_top[0], right_bottom[1]-5), font, 0.5, (255,255,255), 1)
        display_noface_warning -= 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Notice: all the data gathered will be deleted after the demo!", (10,470), font, 0.4, (0,255,0), 1, bottomLeftOrigin=False)
    cv2.putText(frame, f"#Faces: {len(faces_names)}", (10,435), font, 0.4, (255,0,0), 1, bottomLeftOrigin=False)
    cv2.putText(frame, f"Random guess: {(1/len(faces_names))*100:2.1f}%", (10,450), font, 0.4, (255,0,0), 1, bottomLeftOrigin=False)

    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    # cv2.moveWindow("Video", frame.x - 1, frame.y - 1)
    # cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # Display the resulting image
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        if face_names == []:
            display_noface_warning = 50
            continue

        if face_names != [] and (np.array(face_names) == 'Unknown').sum() >= 1 or \
            face_names != [] and (np.array(face_names) != 'Unknown').sum() > 1:
            display_warning_delete = 50
            print("display delete warning")
            continue

        delete_path = f'data/faces/{face_names[-1]}.jpg'
        os.remove(delete_path) 
        print("delete the face: " + delete_path)
        # TODO: do not reload all faces from the file all over again
        load_faces()


    elif key == ord('s'):

        if face_names == []:
            display_noface_warning = 50
            continue

        if face_names != [] and (np.array(face_names) == 'Unknown').sum() > 1 or \
            face_names != [] and (np.array(face_names) != 'Unknown').sum() > 0:
            display_warning = 50
            continue

        print("save the image with name")
        # TODO: find a solution to better integrate the form with the image. (probably having everything in qt or web-based app would be the best)
        import tkinter as tk


        def callback(event):
            # print('e.get():', e1.get())
            # or more universal
            # print('event.widget.get():', event.widget.get())
            # select text after 50ms
            master.after(50, select_all, event.widget)

        def select_all(widget):
            # select text
            widget.select_range(0, 'end')
            # move cursor to the end
            widget.icursor('end')

        def show_entry_fields(event=None):
            print(f"Name: {e1.get()}")
            cv2.imwrite(f"data/faces/{e1.get()}.jpg", original_frame)
            master.destroy()
            load_faces()

        master = tk.Tk()
        master.title('Add Face')
        tk.Label(master,
                 text="Name").grid(row=0,column=0)
        e1 = tk.Entry(master)
        e1.grid(row=0, column=1)
        e1.focus()
        e1.bind('<Control-a>', callback)
        master.bind('<Return>', show_entry_fields) 
        tk.Button(master,
                  text='Enter', command=show_entry_fields).grid(row=0,
                                                               column=2,
                                                               sticky=tk.W,
                                                               pady=4)

        # Gets the requested values of the height and widht.
        windowWidth = master.winfo_reqwidth()
        windowHeight = master.winfo_reqheight()
        # print("Width",windowWidth,"Height",windowHeight)

        # Gets both half the screen width/height and window width/height
        positionRight = int(master.winfo_screenwidth()/2 - windowWidth/2)
        positionDown = int(master.winfo_screenheight()/2 - windowHeight/2)

        # Positions the window in the center of the page.
        master.geometry("+{}+{}".format(positionRight, positionDown))
        tk.mainloop()

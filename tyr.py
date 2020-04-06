import socket
import threading
import logging
import datetime
import os
import struct
import ctypes
import tireddrivinglocal3 as td
import cv2


FORMAT = "%(asctime)s %(threadName)s %(thread)d %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class FaceServer:
    index = 0
    b = '.jpg'
    ad = './image/'
    filename = ad + str(index) + b

    def __init__(self, ip='127.0.0.1', port=9999):
        self.addr = (ip, port)
        self.sock = socket.socket()
        self.clients = {}
        self.event = threading.Event()
        self.checkExsit()

    def checkExsit(self):
        list = os.listdir('./image/')
        for iterm in list:
            iterm = './image/' + iterm
            # print(iterm)
            os.remove(iterm)
        #td.clean()
        print
        'Exsit file has been removed'
        print
        'Create file ...'
        # with open(self.filename, 'wb') as f:
        #     pass

    def start(self):
        self.sock.bind(self.addr)
        self.sock.listen()  # 启动服务
        logging.info('start socket')

        threading.Thread(target=self.accept, name='accept').start()

    def accept(self):
        while not self.event.is_set():
            try:
                s, raddr = self.sock.accept()  # 这里会进入阻塞,所以转到子线程
                #self.checkExsit()
                td.clean()
                self.checkExsit()
                logging.info(s)
                logging.info(raddr)
            except Exception as e:
                logging.error(e)

            self.clients[raddr] = s
            self.save(s, raddr)

    def recvImage(self, sock: socket.socket, addr):
        databuffer = bytes()
        tuichu = False
        while not self.event.is_set():
            # print("pig\n")
            try:

                headsize = 4
                data = sock.recv(1024)  # 接收到的是bytes，阻塞
                # logging.info(data)#打印到日志

                # leng += len(data)
                # index = len(data)
                # datah = data[:8]
                # if index <= 5:
                #     dataed = datal[1024-5+index:1024] + data
                # elif index<=4:
                #     datae = datal[1024-4+index:1024] + data
                #     dataed = datal[1024-4+index:1024] + data
                # else:
                #     datae = data[index-4:index]
                #     dataed = data[index-5:index]

            except Exception as e:
                logging.error(e)

                data = 'quit'
                break

            # print(data)

            # print(dataed)
            # if dataed == self.ALL_END:
            #     self.clients.pop(sock.getpeername())
            #     self.sock.close()
            #     self.index = 0
            #     self.filename = self.ad + str(self.index) + self.b
            #     #self.event.set()
            #     break
            # elif datae == self.PICTURE_PACKAGE_END:
            #     self.index = self.index + 1
            #     self.filename = self.ad + str(self.index) + self.b
            # elif datah == self.PICTURE_PACKAGE_HEAD:
            #     datale = data[9:12]
            #     data = data[12:1024]
            #     lengt = int.from_bytes(datale, byteorder='big', signed=True)
            #     logging.info(lengt)
            #     logging.info(datale)

            if data != 'quit':

                databuffer = databuffer + data
                bodypake = databuffer[:headsize]
                #print(len(databuffer))
                bodysize = int.from_bytes(bodypake, byteorder='big')
                #print(bodysize)
                while True:
                    if len(databuffer) < headsize:
                        break
                    if len(databuffer) < headsize + bodysize:
                        # los = headsize + bodysize - len(databuffer)
                        break

                    body = databuffer[headsize:headsize + bodysize]

                    with open(self.filename, 'ab') as f:
                        f.write(body)
                        f.close()
                        #print(self.filename)
                        #sock.send(self.define(self.filename))
                        t = threading.Thread(target=self.define, args=(sock,self.filename))
                        t.start()
                        t.join()
                        self.index = self.index + 1
                        self.filename = self.ad + str(self.index) + self.b

                        #sock.send(self.define(self.filename))

                    databuffer = databuffer[headsize + bodysize:]

                    if databuffer == b'quit':
                        #sock.send(self.define())
                        self.clients.pop(sock.getpeername())
                        self.sock.close()
                        self.index = 0
                        self.filename = self.ad + str(self.index) + self.b
                        self.event.set()
                        break

            elif data == b'' or data == b'quit':
                #sock.send(self.define())
                logging.info('The message has been all received!')
                self.clients.pop(sock.getpeername())
                self.sock.close()
                self.index = 0
                self.filename = self.ad + str(self.index) + self.b
                self.event.set()
                break
                # 标记记住当前位置
                # length =
            # while True:

            # print(data[index - 2])

            #
            # if data[1] == 1:
            #     logging.info('i have recved one picture')
            #     self.index = self.index + 1
            #     self.filename = self.ad + str(self.index) + self.b
            #     #self.clients.pop(sock.getpeername())
            #     #self.sock.close()
            #     #self.index = 0
            #     #self.filename = self.ad + str(self.index) + self.b
            #     #self.event.set()
            # # elif data == b'':
            # #     print("enmmmmm\n")
            # #     pass

            # elif data[1] == b'\xff' :
            #     logging.info('The message has been all received!')
            #     self.clients.pop(sock.getpeername())
            #     self.sock.close()
            #     self.index = 0
            #     self.filename = self.ad + str(self.index) + self.b
            #     #self.event.set()
            #     break

        print
        'data received'

    def define(self,sock:socket.socket,  filename):


        # road_to_file = "photo/summary8.0"
        # for imgname in os.listdir(road_to_file):
        imgarray = cv2.imread(filename)
        # print(imgarray)
        (a, b, c) = td.Image_evaluate(imgarray)
        # cv2.imshow("Frame", img)

        # print("blinkcount = {0}".format(td.blinkcount))
        print("blink:%d,nod:%d,state:%d" % (a, b, c))
        #a眨眼数 b点头数 c状态
        msg = a.to_bytes(4, byteorder='big', signed=True)  # "".format('-1').encode()
        msg += b.to_bytes(4, byteorder='big', signed=True)
        msg += c.to_bytes(4, byteorder='big', signed=True)
        print(msg)
        sock.send(msg)

    def save(self, sock: socket.socket, addr):

        print
        'Begin to save image ...'

        t = threading.Thread(target=self.recvImage, args=(sock, addr))
        t.start()
        # t.setDaemon(True)
        # t.start()
        # t.join()
        # self.recvImage(sock,addr)

        print
        'Finished saving image ...'

    def raecv(self, sock: socket.socket, addr):
        while not self.event.is_set():
            try:

                data = sock.recv(1024)  # 接收到的是bytes，阻塞
                logging.info(data)  # 打印到日志
            except Exception as e:
                logging.error(e)
            # data = b'quit'

            if data == b'quit':
                self.clients.pop(sock.getpeername())
                sock.close()
                break

            msg = "{} {} {}".format(
                sock.getpeername(),
                datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),
                data.decode()).encode()  # 将发送方的ip和时间戳作为信息发送给每个socket

            for s in self.clients.values():
                s.send(msg)

    def stop(self):

        for s in self.clients.values():
            s.close()
        self.sock.close()
        self.event.set()


cs = FaceServer()
cs.start()

while True:
    cmd = input(">>>")
    if cmd.strip() == 'quit':
        cs.stop()
        threading.Event.wait(3)
        break

    logging.info(threading.enumerate())


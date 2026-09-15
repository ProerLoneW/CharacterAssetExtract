
import os
import sys
import requests
import time
import json
import traceback
from mytools.utils import print_with_timestamp as print
from video_utils import get_video_length, get_video_size, crc64_check, video_reader_check, VideoLoadException




class PipeDataProvider():

    def __init__(self) -> None:
        pass
        

    def prepare(self):
        input_pipe_path = '/data/queue/input'
        output_pipe_path = '/data/queue/output'

        # 确保管道文件存在
        for _ in range(500):
            if os.path.exists(input_pipe_path) and os.path.exists(output_pipe_path):
                break
            else:
                time.sleep(10)
        
        assert os.path.exists(input_pipe_path) and os.path.exists(output_pipe_path)
        self.input_pipe_path = input_pipe_path
        self.output_pipe_path = output_pipe_path


    def process_data(self, operator_function):
        # 打开输入管道文件内容
        with open(self.input_pipe_path, 'r') as input_pipe:
            # 打开输出管道文件内容
            with open(self.output_pipe_path, 'w') as output_pipe:
                while True:
                    input_content = input_pipe.readline()
                    if input_content:
                        try:
                            data = json.loads(input_content)
                            print("读取到的JSON内容:", data)

                            # 检查是否应该结束循环
                            if data.get('end'):
                                print("检测到end为true, 结束程序。")
                                output_pipe.writelines(json.dumps(data) + '\n')
                                output_pipe.flush()
                                break

                            # 添加新的key-value
                            res = data.get('data', data)
                            video_path = os.path.join('/data/file/', res['tos_addr'])
                            print(video_path)
                            # video_path = '/shared_disk/datasets/test_set/0524/2p/zonghe_4_6682.mp4'
                            assert os.path.isfile(video_path)
                            if video_reader_check(video_path) != "ok":
                                continue
                            
                            results = operator_function(video_path, data)
                            if results is None:         # 跳过计算
                                continue

                            self.output_pipe.writelines(json.dumps(results) + '\n')
                            self.output_pipe.flush()


                        except json.JSONDecodeError:
                            print("读取的内容不是有效的JSON:", input_content)
                        except VideoLoadException as e:
                            traceback.print_exc()
                            print("KownException continue")
                        except Exception as e:
                            traceback.print_exc()
                            crc64_check(video_path)
                            print("error {}".format(str(e)))
                            raise e  # 终止程序
                    else:
                        break

    def upload_results(self):
        pass                 


class HttpDataProvider():
    def __init__(self, host=None) -> None:
        if host is None:
            host = "http://localhost:8080"
        self.host = host
        print(self.host)

    
    def prepare(self):
        while True:
            try:
                response = requests.get(self.host + "/ping", timeout=30)
                if response.status_code == 200:
                    break                
            except Exception as e:
                print("In prepare...", str(e))
                pass

            time.sleep(1)
        
    def process_data(self, operator_function):
        while True:
            try:
                response = requests.get(self.host + "/input", timeout=60 * 30)
                if response.status_code != 200:
                    print("服务请求数据出错, ", response.status_code, response)
                    break

                data = response.json()
                print("get data", data)

                # 检查是否应该结束循环
                if data.get('end'):
                    print("检测到end为true, 结束程序。")
                    response = requests.post(self.host + "/output", data=json.dumps(data))
                    break
            
                # 添加新的key-value
                res = data.get('data', data)
                video_path = os.path.join('/data/file/', res['tos_addr'])
                print(video_path)
                # video_path = '/shared_disk/datasets/test_set/0524/2p/zonghe_4_6682.mp4'
                assert os.path.isfile(video_path)
                if video_reader_check(video_path) != "ok":
                    continue

                results = operator_function(video_path, data)
                if results is None:         # 跳过计算
                    continue

                response = requests.post(self.host + "/output", json=results)
                print("set results", response)

            except json.JSONDecodeError:
                print("读取的内容不是有效的JSON:", response.content)
            except VideoLoadException as e:
                traceback.print_exc()
                print("KownException continue")
            except Exception as e:
                traceback.print_exc()
                crc64_check(video_path)
                print("error {}".format(str(e)))
                raise e  # 终止程序

    def upload_results(self, results):
        print(results)
        response = requests.post(self.host + "/output", json=results)
        print("set results", response)   


if __name__ == "__main__":
    provider = HttpDataProvider(host="http://localhost:9991")
    provider.prepare()
    provider.process_data(operator_function=None)
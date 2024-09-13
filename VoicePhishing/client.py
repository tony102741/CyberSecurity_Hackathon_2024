import requests

def upload_audio(file_path):
    url = 'http://localhost:5001/upload'  # 서버 주소
    try:
        with open(file_path, 'rb') as file:
            response = requests.post(url, files={'file': file})
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"요청 오류: {e}")
        return {'error': '서버 요청 오류'}
    except requests.exceptions.JSONDecodeError:
        print("서버 응답에서 JSON 디코딩 오류가 발생했습니다.")
        return {'error': '서버 응답 오류'}

if __name__ == "__main__":
    file_path = 'voice/n2.mp3'  # 업로드할 음성 파일 경로 (상대 경로로 수정)
    result = upload_audio(file_path)
    
    if 'error' in result:
        print(f"오류: {result['error']}")
    else:
        # 단순히 결과만 출력
        print(result['result'])
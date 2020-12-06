from PIL import Image
import os

d = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J', 19: 'K',
                 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
                 30: 'W', 31: 'X', 32: 'Y', 33: 'Z'}

s = {34: '藏', 35: '川', 36: '鄂', 37: '甘', 38: '赣', 39: '贵', 40: '桂', 41: '黑', 42: '沪', 43: '吉',
                 44: '冀', 45: '津', 46: '晋', 47: '京', 48: '辽', 49: '鲁', 50: '蒙', 51: '闽', 52: '宁', 53: '青', 54: '琼',
                 55: '陕', 56: '苏', 57: '皖', 58: '湘', 59: '新', 60: '渝', 61: '豫', 62: '粤', 63: '云', 64: '浙'}
def StrtoLabel(Str):
    label = []
    for i in range(0, 7):
        if 48 <= ord(Str[i]) <= 57:  # 0-9
            # print(ord(Str[i]))  # 1通过ascll转为49
            # print(ord('0'))  # 0通过ascll转为48
            label.append(ord(Str[i]) - ord('0'))

        elif 65 <= ord(Str[i]) <= 90:  # A-Z

            for k, v in d.items():
                if Str[i] == v:
                    out = k
                    label.append(out)
        else:
            # ['藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京',
            #  '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫',
            #  '粤', '云', '浙']

            for k, v in s.items():
                if Str[i] == v:
                    out = k
                    # print(type(out))
                    label.append(out)
    # print(label)
    return label


def trans_square(image):  # 将图片转成正方形
    r"""Open the image using PIL."""
    image = image.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(0, 0, 0))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    background = background.resize((224, 224))

    return background


def LabeltoStr(Label):
    Str = ""
    for i in Label:

        if i <= 9:
            Str += chr(ord('0') + i)
        elif 10 <= i <= 33:
            for k, v in d.items():
                if i == k:
                    out = v
                    Str += out
        else:
            for k, v in s.items():
                if i == k:
                    out = v
                    Str += out

    return Str


if __name__ == '__main__':
    main_path = r"blue_plate"
    label_data = StrtoLabel("桂4F4HJT")
    # StrtoLabel("0AZ川12D")

    Str_data = LabeltoStr(label_data)
    print(Str_data)




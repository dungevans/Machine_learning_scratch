import numpy as np 
import torch 
import torch.nn as nn

class utils : 

    def __init__ ( self, algorithm : str ) : 
        self.algorithm = algorithm
    def pre_tokenization ( self, input : torch.tensor )-> torch.tensor : 
        """
        Nhiệm vụ: Đọc văn bản thô, chia thành các từ cơ bản, thêm ký tự kết thúc từ </w> để đánh dấu ranh giới, 
        băm tất cả thành từng chữ cái riêng lẻ và đếm tần suất của chúng. 
        Thay vì thao tác trên toàn bộ văn bản dài, ta nén nó lại thành một cuốn "từ điển tần suất" để tối ưu hiệu năng.
        """
        dictionary = []
        words = input.split()
        for sub in words :
            dictionary.append ( sub.split() ) 






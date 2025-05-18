from pydantic import BaseModel 
class get_description(BaseModel):
        description:str 
        title:str
class get_info(BaseModel):
        posts:list
        likes:list 
        comments:list
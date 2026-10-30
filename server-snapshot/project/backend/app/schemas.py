from pydantic import BaseModel, Field, model_validator


class Parameters(BaseModel):
    length: float = Field(default=80, ge=50, le=200)
    width: float = Field(default=60, ge=40, le=160)
    thickness: float = Field(default=12, ge=5, le=40)
    hole: float = Field(default=20, ge=4, le=60)

    @model_validator(mode='after')
    def fits(self):
        if self.hole > min(self.length, self.width) - 28:
            raise ValueError('中心孔过大，请至少保留 14 mm 边距')
        return self


class MessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class RunInput(BaseModel):
    parameters: Parameters | None = None

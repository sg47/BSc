import itertools
import cv2  
import numpy as np

class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1<x2 and y1<y2:
            return type(self)(x1, y1, x2, y2)
    __and__ = intersection

    def difference(self, other):
        inter = self&other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1<other.x1<self.x2: xs.add(other.x1)
        if self.x1<other.x2<self.x2: xs.add(other.x2)
        if self.y1<other.y1<self.y2: ys.add(other.y1)
        if self.y1<other.y2<self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
            pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect!=inter:
                yield rect
    __sub__ = difference

    def __init__(self, x1, y1, x2, y2):
        if x1>x2 or y1>y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self)==tuple(other)
    def __ne__(self, other):
        return not (self==other)

    def __repr__(self):
        return type(self).__name__+repr(tuple(self))

    def upper_left(self):
        return (int(self.x1) , int(self.y1))
    def bottom_right(self):
        return (int(self.x2) , int(self.y2))


def pairwise(iterable):
    # https://docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

if __name__ == '__main__':
    main()

def main():
    # black blank image
    # blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    # print(blank_image.shape)
    # cv2.imshow("Black Blank", blank_image)



    # white blank image
    blank_image2 = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)

    # 1.
    a = Rectangle(50, 50, 150, 150)
    cv2.rectangle(blank_image2, a.upper_left(), a.bottom_right(), (0, 255, 0), 5)

    # b = Rectangle(50, 50, 150, 150)
    # c = a&b
    # print(c)
    # # Rectangle(0.5, 0.5, 1, 1)
    # # print(list(b-a))
    # # [Rectangle(0, 0, 0.5, 0.5), Rectangle(0, 0.5, 0.5, 1), Rectangle(0.5, 0, 1, 0.5)]
    # cv2.rectangle(blank_image2, b.upper_left(), b.bottom_right(), (0, 255, 0), 5)
    # cv2.rectangle(blank_image2, c.upper_left(), c.bottom_right(), (0, 0, 255), -1)

    # crop_img = blank_image2[c.y1:c.y2, c.x1:c.x2]
    # print('mean: ', np.mean(crop_img))

    # cv2.imshow("White Blank", blank_image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # 2.
    # b = Rectangle(25, 25, 125, 75)
    # c = a&b
    # print(c)
    # # Rectangle(0.25, 0.25, 1, 0.75)
    # # print(list(a-b))
    # # [Rectangle(0, 0, 0.25, 0.25), Rectangle(0, 0.25, 0.25, 0.75), Rectangle(0, 0.75, 0.25, 1), Rectangle(0.25, 0, 1, 0.25), Rectangle(0.25, 0.75, 1, 1)]
    # cv2.rectangle(blank_image2, b.upper_left(), b.bottom_right(), (0, 255, 0), 5)
    # cv2.rectangle(blank_image2, c.upper_left(), c.bottom_right(), (0, 0, 255), -1)

    # crop_img = blank_image2[c.y1:c.y2, c.x1:c.x2]
    # print('mean: ', np.mean(crop_img))

    # cv2.imshow("White Blank", blank_image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # # 3.
    # b = Rectangle(25, 25, 75, 75)
    # c = a&b
    # print(c)
    # Rectangle(0.25, 0.25, 0.75, 0.75)
    # print(list(a-b))
    # [Rectangle(0, 0, 0.25, 0.25), Rectangle(0, 0.25, 0.25, 0.75), Rectangle(0, 0.75, 0.25, 1), Rectangle(0.25, 0, 0.75, 0.25), Rectangle(0.25, 0.75, 0.75, 1), Rectangle(0.75, 0, 1, 0.25), Rectangle(0.75, 0.25, 1, 0.75), Rectangle(0.75, 0.75, 1, 1)]
    # print(b.upper_left())
    # cv2.rectangle(blank_image2, b.upper_left(), b.bottom_right(), (0, 255, 0), 5)
    # cv2.rectangle(blank_image2, c.upper_left(), c.bottom_right(), (0, 0, 255), -1)

    # crop_img = blank_image2[c.y1:c.y2, c.x1:c.x2]
    # print('mean: ', np.mean(crop_img))

    # cv2.imshow("White Blank", blank_image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # 4.
    # b = Rectangle(150, 150, 200, 200)
    # cv2.rectangle(blank_image2, b.upper_left(), b.bottom_right(), (0, 255, 0), 5)

    # if a&b is not None:
    #     c = a&b
    #     print(c)
    # # None
    # # print(list(a-b))
    # # [Rectangle(0, 0, 1, 1)]
    #     print(b.upper_left())
    #     cv2.rectangle(blank_image2, c.upper_left(), c.bottom_right(), (0, 0, 255), -1)
    #     crop_img = blank_image2[c.y1:c.y2, c.x1:c.x2]
    #     print('mean: ', np.mean(crop_img))

    # cv2.imshow("White Blank", blank_image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # 5.
    b = Rectangle(10, 10, 300, 300)
    print(a&b)
    # Rectangle(0, 0, 1, 1)
    print(list(a-b))
    # []
    cv2.rectangle(blank_image2, b.upper_left(), b.bottom_right(), (255, 0, 0), 5)

    if a&b is not None:
        c = a&b
        print(c)
    # None
    # print(list(a-b))
    # [Rectangle(0, 0, 1, 1)]
        print(b.upper_left())
        cv2.rectangle(blank_image2, c.upper_left(), c.bottom_right(), (0, 0, 255), -1)
        crop_img = blank_image2[c.y1:c.y2, c.x1:c.x2]
        print('mean: ', np.mean(crop_img))

    cv2.imshow("White Blank", blank_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
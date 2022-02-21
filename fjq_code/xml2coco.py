import xml.sax
import os
import json

groups = {'Ship': ['Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'other-ship',
                   'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship'],
          'Vehicle': ['Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'other-vehicle',
                      'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor'],
          'Airplane': ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21',
                       'C919', 'A220', 'A321', 'A330', 'A350', 'other-airplane'],
          'Court': ['Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court'],
          'Road': ['Roundabout', 'Intersection', 'Bridge']}


class GetAnns(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ''
        self.label = ''
        self.points = []
        self.temp_point = ''
        self.count = 0
        self.shapes = []
        self.shape = {}
        self.content_stack = ''

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == 'object':
            self.count += 1
            if self.count == 158:
                print('Here!')
            print(self.count, 'Get an object!')
        self.content_stack = ''

    def endElement(self, tag):
        if tag == 'object':
            self.points = []
            self.shapes.append(self.shape)
            self.shape = {}

        if self.CurrentData == 'name':
            self.label = self.content_stack
            self.shape['label'] = self.label
            for group in groups.keys():
                if self.content_stack in groups[group]:
                    self.shape['group'] = group
        elif self.CurrentData == 'point':
            point = self.content_stack.split(',')
            point = list(map(float, point))
            self.points.append(point)
            if len(self.points) == 6:
                print('{} Error!\n'.format(self.count))
                log.write('{} Error!\n'.format(self.count))
                print(self.points)
                log.write('{}\n'.format(self.points))
            if len(self.points) == 5:
                for p in self.points:
                    if len(p) != 2:
                        print(self.count, self.points.index(p), 'Error!')
                        log.write('{} {} Error!\n'.format(self.count, self.points.index(p)))
                del self.points[-1]
                self.shape['points'] = self.points

        self.CurrentData = ''

    def characters(self, content):
        self.content_stack = self.content_stack + content


if __name__ == '__main__':
    # 人工定义待写入json的全局变量
    image_list= []
    category_list= []
    annotation_list= []
    ann_total = {'images': image_list, 'categories': category_list, 'annotations': annotation_list}
    # 标注文件xml 和 待生成json文件的位置
    xml_path = '/home/fengjq/workspace/xtb_dataset/train/labelXml/'
    save_path = '/home/fengjq/workspace/xtb_dataset/train/'

    xml_list = os.listdir(xml_path)
    num = 0
    for xml_name in xml_list:
        if '.xml' in xml_name:
            num += 1
            print(str(num) + " Processing " + xml_name)
            # 开始读取xml文件
            xml_file = xml_path + xml_name
            parser = xml.sax.make_parser()  # create XMLReader
            parser.setFeature(xml.sax.handler.feature_namespaces, 0)  # turn off namespaces
            Handler = GetAnns()  # rewrite ContextHandler
            parser.setContentHandler(Handler)
            parser.parse(xml_file)
            
            ann = {'shapes': Handler.shapes}
            ann_json = json.dumps(ann, indent=4)
            with open(save_file, 'w') as json_f:
                json_f.write(ann_json)
            print('Over!')
            

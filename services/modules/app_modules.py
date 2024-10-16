import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from services.modules.inference_modules import crop_image, detection_onnx_predict, classify_bboxes, resize_with_padding


class AppFunctions:
    def __init__(self, classify_model, detection_model, classify_imgsz, detection_imgw):
        self.detection_model = detection_model
        self.detection_imgw = detection_imgw
        self.classify_model = classify_model
        self.classify_imgsz = classify_imgsz

        self.class_names = eval(self.classify_model._model_meta.custom_metadata_map['names'])
        self.destroyed_classes = [
            'disintegrated',
            'mashed',
            'moldy',
            'nightingale_eaten',
            ]

        self.current_index = 0

    def process_images(self, image_1, image_2):
        self.img_1 = crop_image(image_1, self.detection_imgw)
        self.img_2 = crop_image(image_2, self.detection_imgw)

        detection_result_1 = detection_onnx_predict(self.img_1.copy(), self.detection_imgw, self.detection_model)
        detection_result_2 = detection_onnx_predict(self.img_2.copy(), self.detection_imgw, self.detection_model)

        classify_result_1 = classify_bboxes(detection_result_1, self.classify_model, self.img_1.copy(), "", self.detection_imgw, self.classify_imgsz, self.class_names)
        classify_result_2 = classify_bboxes(detection_result_2, self.classify_model, self.img_2.copy(), "", self.detection_imgw, self.classify_imgsz, self.class_names)
        
        self.bboxes_1, self.croped_images_1, self.pr_class_names_1, self.colors_1, self.sizes_1, result_1 = classify_result_1
        self.bboxes_2, self.croped_images_2, self.pr_class_names_2, self.colors_2, self.sizes_2, result_2 = classify_result_2

        dates_num_1 = len(self.bboxes_1)
        dates_num_2 = len(self.bboxes_2)
        dates_num = max(dates_num_1, dates_num_2)

        destroyed_list_1 = [class_name in self.destroyed_classes for class_name in self.pr_class_names_1]
        destroyed_num_1 = destroyed_list_1.count(True)
        destroyed_list_2 = [class_name in self.destroyed_classes for class_name in self.pr_class_names_2]
        destroyed_num_2 = destroyed_list_2.count(True)

        destroyed_num = 0
        for idx in range(dates_num):
            if destroyed_list_1[idx] or destroyed_list_2[idx]:
                destroyed_num += 1


        destroyed_percent_1 = destroyed_num_1 / dates_num_1 * 100
        destroyed_percent_2 = destroyed_num_2 / dates_num_2 * 100
        destroyed_percent = destroyed_num / dates_num * 100

        mean_length = sum(self.sizes_1[0]) / len(self.sizes_1[0])
        mean_width = sum(self.sizes_1[1]) / len(self.sizes_1[1])

        self.results_1 = cv2.cvtColor(result_1, cv2.COLOR_RGB2BGR)
        self.results_2 = cv2.cvtColor(result_2, cv2.COLOR_RGB2BGR)
        nav_img_1 = cv2.cvtColor(self.img_1.copy(), cv2.COLOR_RGB2BGR)
        nav_img_2 = cv2.cvtColor(self.img_2.copy(), cv2.COLOR_RGB2BGR)
        bbox_1 = self.bboxes_1[self.current_index]
        bbox_2 = self.bboxes_2[self.current_index]

        nav_result_1 = cv2.rectangle(nav_img_1, (bbox_1[:2]), (bbox_1[2:]), (0, 255, 0), 5)
        nav_result_2 = cv2.rectangle(nav_img_2, (bbox_2[:2]), (bbox_2[2:]), (0, 255, 0), 5)

        dict_of_classes = {x:self.pr_class_names_1.count(x) for x in self.pr_class_names_1}

        rainbow_colors = ["#" + ''.join([hex(int(255 * i))[2:].zfill(2) for i in [r, g, b]]) 
                  for r, g, b, _ in [plt.cm.rainbow(x) for x in np.linspace(0, 1, len(dict_of_classes.keys()))]]
        
        fig = go.Figure(data=[go.Bar(x=list(dict_of_classes.keys()), y=list(dict_of_classes.values()))])
        fig.update_layout(bargap=0.2, bargroupgap=0.1)
        fig.data[0].marker.color = rainbow_colors[:len(dict_of_classes)]

        pr_class_name_1 = self.pr_class_names_1[self.current_index]
        pr_class_name_2 = self.pr_class_names_2[self.current_index]

        color_1 = self.colors_1[self.current_index]
        color_2 = self.colors_2[self.current_index]

        size_1 = self.sizes_1[self.current_index]
        length_1, width_1 = size_1[0], size_1[1]

        size_2 = self.sizes_2[self.current_index]
        length_2, width_2 = size_2[0], size_2[1]

        date_1 = cv2.cvtColor(self.croped_images_1[self.current_index], cv2.COLOR_RGB2BGR)
        date_1 = resize_with_padding(date_1, (500, 500))
        date_2 = cv2.cvtColor(self.croped_images_2[self.current_index], cv2.COLOR_RGB2BGR)
        date_2 = resize_with_padding(date_2, (500, 500))


        return (
            self.results_1, self.results_2, 
            dates_num_1, dates_num_2, dates_num, 
            destroyed_percent_1, destroyed_percent_2, destroyed_percent, 
            destroyed_num_1, destroyed_num_2, destroyed_num,
            mean_length, mean_width, fig,
            pr_class_name_1, pr_class_name_2, 
            color_1, color_2, 
            length_1, width_1, 
            length_2, width_2, 
            date_1, date_2,
            nav_result_1, nav_result_2
            )

    def next_date(self):
        if self.current_index < len(self.bboxes_1) - 1:
            self.current_index +=1
        else:
            self.current_index = 0

        pr_class_name_1 = self.pr_class_names_1[self.current_index]
        pr_class_name_2 = self.pr_class_names_2[self.current_index]

        color_1 = self.colors_1[self.current_index]
        color_2 = self.colors_2[self.current_index]

        size_1 = self.sizes_1[self.current_index]
        length_1, width_1 = size_1[0], size_1[1]

        size_2 = self.sizes_2[self.current_index]
        length_2, width_2 = size_2[0], size_2[1]

        date_1 = cv2.cvtColor(self.croped_images_1[self.current_index], cv2.COLOR_RGB2BGR)
        date_1 = resize_with_padding(date_1, (500, 500))
        date_2 = cv2.cvtColor(self.croped_images_2[self.current_index], cv2.COLOR_RGB2BGR)
        date_2 = resize_with_padding(date_2, (500, 500))

        bbox_1 = self.bboxes_1[self.current_index]
        bbox_2 = self.bboxes_2[self.current_index]
        nav_img_1 = cv2.cvtColor(self.img_1.copy(), cv2.COLOR_RGB2BGR)
        nav_img_2 = cv2.cvtColor(self.img_2.copy(), cv2.COLOR_RGB2BGR)

        results_1 = cv2.rectangle(nav_img_1, (bbox_1[:2]), (bbox_1[2:]), (0, 255, 0), 5)
        results_2 = cv2.rectangle(nav_img_2, (bbox_2[:2]), (bbox_2[2:]), (0, 255, 0), 5)

        
        return (
            pr_class_name_1, pr_class_name_2, 
            color_1, color_2, 
            length_1, width_1, 
            length_2, width_2, 
            date_1, date_2,
            results_1, results_2
            )

    def previous_date(self):
        if self.current_index > 0:
            self.current_index -= 1
        else:
            self.current_index = len(self.bboxes_1) - 1

        pr_class_name_1 = self.pr_class_names_1[self.current_index]
        pr_class_name_2 = self.pr_class_names_2[self.current_index]

        color_1 = self.colors_1[self.current_index]
        color_2 = self.colors_2[self.current_index]

        size_1 = self.sizes_1[self.current_index]
        length_1, width_1 = size_1[0], size_1[1]

        size_2 = self.sizes_2[self.current_index]
        length_2, width_2 = size_2[0], size_2[1]

        date_1 = cv2.cvtColor(self.croped_images_1[self.current_index], cv2.COLOR_RGB2BGR)
        date_1 = resize_with_padding(date_1, (500, 500))
        date_2 = cv2.cvtColor(self.croped_images_2[self.current_index], cv2.COLOR_RGB2BGR)
        date_2 = resize_with_padding(date_2, (500, 500))

        bbox_1 = self.bboxes_1[self.current_index]
        bbox_2 = self.bboxes_2[self.current_index]
        nav_img_1 = cv2.cvtColor(self.img_1.copy(), cv2.COLOR_RGB2BGR)
        nav_img_2 = cv2.cvtColor(self.img_2.copy(), cv2.COLOR_RGB2BGR)

        results_1 = cv2.rectangle(nav_img_1, (bbox_1[:2]), (bbox_1[2:]), (0, 255, 0), 5)
        results_2 = cv2.rectangle(nav_img_2, (bbox_2[:2]), (bbox_2[2:]), (0, 255, 0), 5)

        return (
            pr_class_name_1, pr_class_name_2, 
            color_1, color_2, 
            length_1, width_1, 
            length_2, width_2, 
            date_1, date_2,
            results_1, results_2
            )
    

def app(app_functions, args):
    css = """
        .gr-process-button {
            color: #007bff;
            font-size: 38px;
            width: 200px;
        }
        .gr-submit-button {
            color: orange;
            font-size: 28px;
        }
        .nav-image {
        margin: auto;
        display: flex;
        justify-content: center;
        height: 500px;
        }
    """
    
    with gr.Blocks(css=css, title="Inference App") as demo:
        gr.HTML("""
                <div style='text-align:center; background-color: #007bff; font-size: 30px;color: white; padding: 10px'>
                    Ez Dates Classification
                </div>
                """)
        gr.HTML("""<h2 style='text-align:center; padding-top: 50px;'>Images</h2>""")
        with gr.Row():
            image_1 = gr.Image(label="Front")
            image_2 = gr.Image(label="Back")
            result_1 = gr.Image(label="Result 1")
            result_2 = gr.Image(label="Result 2")

        with gr.Row():
            process_button = gr.Button("Process", elem_classes="gr-process-button")

        gr.HTML("""<hr style='height:2px;border-width:0;color:gray;background-color:gray'>""")
        with gr.Row():
            with gr.Column():
                gr.HTML("""<h2 style='text-align:center;'>Front</h2>""")
                with gr.Row():
                    dates_num_1 = gr.Textbox(label="Number of dates")
                with gr.Row():
                    destroyed_percent_1 = gr.Textbox(label="Percentage of destroyed")
                with gr.Row():
                    destroyed_num_1 = gr.Textbox(label="Number of destroyed dates")

            with gr.Column():
                gr.HTML("""<h2 style='text-align:center;'>Back</h2>""")
                with gr.Row():
                    dates_num_2 = gr.Textbox(label="Number of dates")
                with gr.Row():
                    destroyed_percent_2 = gr.Textbox(label="Percentage of destroyed")
                with gr.Row():
                    destroyed_num_2 = gr.Textbox(label="Number of destroyed dates")

            with gr.Column():
                gr.HTML("""<h2 style='text-align:center;'>Result</h2>""")
                with gr.Row():
                    dates_num = gr.Textbox(label="Number of dates")
                with gr.Row():
                    destroyed_percent = gr.Textbox(label="Percentage of destroyed")
                with gr.Row():
                    destroyed_num = gr.Textbox(label="Number of destroyed dates")

        gr.HTML("""<hr style='height:2px;border-width:0;color:gray;background-color:gray'>""")
        gr.HTML("""<h2 style='text-align:center;'>Measurements</h2>""")

        with gr.Row():
            mean_length = gr.Textbox(label="Mean length")
            mean_width = gr.Textbox(label="Mean width")

        with gr.Row():
            bar_chart = gr.Plot()

        gr.HTML("""<hr style='height:2px;border-width:0;color:gray;background-color:gray'>""")
        gr.HTML("""<h2 style='text-align:center;'>Dates</h2>""")

        
        with gr.Row(elem_classes='nav-image'):
            nav_result_1 = gr.Image(label="Front",elem_classes='nav-image')
            date_image_front = gr.Image(label="Dates front")
            nav_result_2 = gr.Image(label="Back",elem_classes='nav-image')
            date_image_back = gr.Image(label="Dates Back")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    date_color_front = gr.Textbox(label="Date color")
                    type_of_destroyed_front = gr.Textbox(label="Type of destroyed")
                with gr.Row():
                    date_length_front = gr.Textbox(label="Date length")
                    date_width_front = gr.Textbox(label="Date width")
            with gr.Column(scale=2):
                with gr.Row():
                    date_color_back = gr.Textbox(label="Date color")
                    type_of_destroyed_back = gr.Textbox(label="Type of destroyed")
                with gr.Row():
                    date_length_back = gr.Textbox(label="Date length")
                    date_width_back = gr.Textbox(label="Date width")

        with gr.Row():
            previous_date_button = gr.Button("Previous", elem_classes='gr-submit-button')
            next_date_button = gr.Button("Next", elem_classes='gr-submit-button')


        process_button.click(
            fn=app_functions.process_images, 
            inputs=[image_1, image_2], 
            outputs=[
                result_1, result_2, 
                dates_num_1, dates_num_2, dates_num, 
                destroyed_percent_1, destroyed_percent_2, destroyed_percent, 
                destroyed_num_1, destroyed_num_2, destroyed_num,
                mean_length, mean_width, bar_chart,
                type_of_destroyed_front, type_of_destroyed_back, 
                date_color_front, date_color_back,
                date_length_front, date_width_front,
                date_length_back, date_width_back,
                date_image_front, date_image_back,
                nav_result_1, nav_result_2
                ])
 
        previous_date_button.click(
            fn=app_functions.previous_date, 
            outputs=[
                type_of_destroyed_front, type_of_destroyed_back, 
                date_color_front, date_color_back,
                date_length_front, date_width_front,
                date_length_back, date_width_back,
                date_image_front, date_image_back,
                nav_result_1, nav_result_2
                ])
        
        next_date_button.click(
            fn=app_functions.next_date, 
            outputs=[
                type_of_destroyed_front, type_of_destroyed_back, 
                date_color_front, date_color_back,
                date_length_front, date_width_front,
                date_length_back, date_width_back,
                date_image_front, date_image_back,
                nav_result_1, nav_result_2
                ])

    demo.launch(server_port=args.port, share=True)

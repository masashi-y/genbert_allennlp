import sys
import argparse
import json
import jsonlines


header = '''\
<html>
<head>
    <meta charset="UTF-8">
    <style type='text/css'>
        .arith_span {
            background-color: #ffccff;
        }
    </style>
    <link href="https://unpkg.com/tabulator-tables@4.6.3/dist/css/tabulator.min.css" rel="stylesheet">
    <script type="text/javascript" src="https://unpkg.com/tabulator-tables@4.6.3/dist/js/tabulator.min.js"></script>
<head>
<body>
<div>
    <button id="download-json">Download JSON</button>
</div>
<div id="drop-table"></div>
<script>
var tabledata = \
'''

footer = '''\
;

tabledata.forEach(function(item, index) {

    item.passage_text = "";
    var offset = item.question_tokens.length + 2;
    for (var token_index = 0; token_index < item.passage_tokens.length; token_index++) {
        var token_text = item.passage_tokens[token_index],
            is_wordpieced = token_text.startsWith("##"),
            is_predicted_argument = "predicted_argument_indices" in item
                && item.predicted_argument_indices.includes(offset + token_index);
        
        if (is_wordpieced) {
            token_text = token_text.substr(2);
        }

        item.passage_text +=
            (is_wordpieced ? "" : " ")
            + (is_predicted_argument ? "<span class=\\"arith_span\\">" + token_text + "</span>" : token_text);
    }

    this[index] = item;
}, tabledata);

function format_passage(cell) {
    return cell.getValue();
    // return cell.getValue().join(' ').replace(/ ##/g, '');
}

function format_question(cell) {
    return cell.getValue().join(' ').replace(/ ##/g, '');
}

function format_calculator(cell) {
    let calculator_outputs = cell.getValue();
    if (calculator_outputs) {
        console.log(calculator_outputs);
        if (calculator_outputs.predicted_tokens.length == 0)
            return "<b>∅</b>";
        let output = calculator_outputs.predicted_tokens.join(' ').replace(/ ##/g, '');
        return output;
    } else {
        return "";
    }
}

function format_true_label(cell) {
    let answer_annotation = cell.getValue()[0];
    if (answer_annotation.number) {
        return answer_annotation.number;
    } else if (answer_annotation.spans) {
        return answer_annotation.spans[0];
    } else {
        console.log(answer_annotation);
        return answer_annotation;
    }
}

var table = new Tabulator("#drop-table", {
    data: tabledata,
    height: "98%",
    layout:"fitDataFill",
    responsiveLayout:"collapse",
    pagination: "local",
    paginationSize: 100,
    selectable: false,
    paginationSizeSelector: [10, 20, 50, 100],
    columns:[
        {formatter:"rownum", hozAlign:"center", width:40, topCalc: "max"},
        {title:"Id", field:"question_id", width:100, responsive:0, editor: "input"},
        {title:"Ans span", field:"answer.span", width:250, responsive:0, editor: true},
        {title:"Ans gen.", field:"answer.generation", width:250, responsive:0, editor: true},
        {title:"Ans type", field:"answer.answer_type", width:250, responsive:0, editor: true},
        {title:"Question", field:"question_tokens", formatter: format_question},
        {title:"Passage", field:"passage_text", formatter: format_passage},
        {title:"Calculator", field:"calculator_outputs", formatter: format_calculator},
        {title:"True label", field:"answer_annotations", formatter: format_true_label},
    ],
});

document.getElementById("download-json").addEventListener("click", function() {
    table.download("json", "data.json");
});

</script>
</body>
</html>\
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="output a json file for visualization"
    )

    parser.add_argument(
        "TARGET",
        help="output.json file"
    )

    args = parser.parse_args()

    results = list(jsonlines.open(args.TARGET))

    print(header)
    json.dump(results, sys.stdout, sort_keys=True, indent=4)
    print(footer)

/*
 * @Author: JasonYU
 * @Date: 2020-10-01 10:16:39
 * @LastEditTime: 2020-10-04 10:05:01
 * @FilePath: \SE\flask_se\frontend\js\index.js
 */
$(document).ready(function () {
    
    $("#resultSection").hide();

    
    $("#searchPaper").click(function () {
        keyword = $("#keyword").val();
        searchPaper(keyword);
    });

    
    $("#keyword").on("input propertychange", function () {
        getSuggest();
    });
});


$(document).keyup(function (event) {
    if (event.keyCode == 13) {
        searchPaper($("#keyword").val());
    }
});

function searchPaper(key) {
    
    $("#result").empty();
    $.getJSON({
        url: "http://localhost:4000/search/" + key,
        success: function (result) {
            
            console.log(result.docs)
            res = result.docs;
            
            if (res.length) {
                for (i = 0; i < res.length; i++) {
                    
                    resultItem =
                        `
                        <div class='col-md-12 mb-4'>
                            <div class='card mb-12 shadow-sm'>
                                <div class='card-body'>
                                    <h5>` + res[i].title + ` <small style='margin-left: 10px'>` + res[i].authors + `</small> <small style='margin-left: 10px'>` + res[i].year +  `</small></h5>
                                    <p class='card-text'>` + res[i].abstract + `</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    $("#result").append(resultItem);
                }

                $("section.jumbotron").animate({
                    margin: "0"
                });
                
                $("#resultSection").show();
                
                $("#suggestList").empty();
            }
        }
    });
}

============
Contributors
============

Thanks to the following users for their contributions to Pyramid!

.. raw:: html

    <!-- Block section -->
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <script type="text/javascript">
        function commaFmt(x) {
            return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        }

        $(document).ready(function() {
            $.getJSON("https://api.github.com/repos/tgsmith61591/pyramid/stats/contributors", function(arr) {
                // sort the array based on total count
                arr.sort(function(a, b) {
                    var aTotal = a['total'];
                    var bTotal = b['total'];

                    // reverse for desc
                    return (aTotal > bTotal) ? -1 : (bTotal > aTotal) ? 1 : 0;
                });

                $.each(arr, function(i, obj) {
                    var total = obj['total'];
                    var adds = 0;
                    var dels = 0;

                    // get the counts of adds/deletes
                    $.each(obj['weeks'], function(wk, weekData) {
                        adds += weekData['a'];
                        dels += weekData['d'];
                    });

                    var authorJSON = obj['author'];
                    var authorLogin = authorJSON['login'];
                    var authorURL = authorJSON['html_url'];
                    var avatarURL = authorJSON['avatar_url'] + '&s=60';

                    // Add HTML elements to the ol element below
                    var li = $('<li class="capped-card">' +
                                    '<h3>' +
                                        '<img src="' + avatarURL + '" class="avatar" alt="">' +
                                        '<span class="rank">#' + (i+1).toString() + '</span>' +
                                        '<a class="aname" href="' + authorURL + '">' + authorLogin + '</a>' +
                                        '<span class="ameta">' +
                                            '<span class="cmeta">' +
                                                '<a href="https://github.com/tgsmith61591/pyramid/commits?author=' + authorLogin + '" class="cmt">' + total.toString() + ' commit' + (total > 1 ? 's' : '') + '</a>' +
                                                '/' +
                                                '<span class="a">' + commaFmt(adds) + ' ++</span>' +
                                                '/' +
                                                '<span class="d">' + commaFmt(dels) + ' --</span>' +
                                            '</span>' +
                                        '</span>' +
                                    '</h3>' +
                                '</li>');

                    $('#contrib').append(li);
                });
            });
        });
    </script>

    <!-- This is taken from the Github contrib page -->
    <ol id="contrib" class="contrib-data capped-cards clearfix"></ol>
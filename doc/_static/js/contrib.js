function commaFmt(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function fetchContributors() {
    $.getJSON("https://api.github.com/repos/alkaline-ml/pmdarima/stats/contributors", function(arr) {
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
            var p = (total > 1) ? 's' : '';

            // Add HTML elements to the ol element below
            var li = $('<li class="capped-card" style="display; block">' +
                         '<div class="contrib-wrapper">' +
                           '<div class="contrib-avatar-wrapper">' +
                             '<img src="' + avatarURL + '" class="avatar" alt="">' +
                           '</div>' +
                           '<div class="contrib-author-wrapper">' +
                             '<h3><a class="committer" href="' + authorURL + '">' + authorLogin + '</a></h3>' +
                           '</div>' +
                           '<div class="contrib-rank-wrapper">' +
                             '<span class="rank">#' + (i + 1).toString() + '</span>' +
                           '</div>' +
                           '<div class="contrib-stats-wrapper">' +
                             '<span class="ameta">' +
                               '<span class="cmeta">' +
                                 '<a href="https://github.com/alkaline-ml/pmdarima/commits?author=' + authorLogin + '" class="cmt">' + commaFmt(total) + ' commit' + p + '</a> / ' +
                                 '<span class="a">' + commaFmt(adds) + ' ++</span> / ' +
                                 '<span class="d">' + commaFmt(dels) + ' --</span>' +
                               '</span>' +
                             '</span>' +
                           '</div>' +
                         '</div>' +
                        '</li>')

            // can only do this once the doc is ready
            $('#contrib').append(li);
        });
    });
}

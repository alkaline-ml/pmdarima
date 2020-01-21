.. _citing:

======
Citing
======

If you would like to include ``pmdarima`` in your published work, please cite it as follows:

.. raw:: html

    <ul>
      <li>Smith, Taylor G., <i>et al.</i> pmdarima: ARIMA estimators for Python, 2017-,
      <a href=http://www.alkaline-ml.com/pmdarima target="_blank">http://www.alkaline-ml.com/pmdarima</a>
      [Online; accessed

      <!-- So we can have the current date in the pre-written citation -->
        <script type="text/javascript">
          var today = new Date();
          var formattedDate = [
            today.getFullYear(),
            ('0' + (today.getMonth() + 1)).slice(-2),
            ('0' + today.getDate()).slice(-2),
          ].join('-');
          document.write(formattedDate);
          document.write('].'); // Easier to just put this in the script tag
        </script>

      </li>
    </ul>

BibTeX Entry:

.. code-block:: tex

    @MISC {pmdarima,
      author = {Taylor G. Smith and others},
      title  = {{pmdarima}: ARIMA estimators for {Python}},
      year   = {2017--},
      url    = "http://www.alkaline-ml.com/pmdarima",
      note   = {[Online; accessed <today>]}
    }
